"""
Amazon Robotics Hackathon - Routing API

This module contains a deterministic, rolling-horizon fleet planner. The
planner deliberately uses only the Python standard library so that the file
can be submitted on its own.

*****IMPORTANT*****
Team name: jom
Email address: jiratnimstudent@ubc.ca
*******************
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ar_hackathon.models.graph_state import GraphState


INF = float("inf")

# Everything below is a cache or a plan; the authoritative state always comes
# from the GraphState supplied by the engine.
_graph_signature: Optional[Tuple] = None
_adjacency: Dict[int, List[Tuple[int, object]]] = {}
_distances: Dict[int, Dict[int, float]] = {}
_actions_time: Optional[int] = None
_actions: Dict[int, Optional[int]] = {}
_last_time_seen: Optional[int] = None
_pending_calls = set()
_forecast = {}
_seen_pods = set()
_arrival_history = {}


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """Return a coordinated next hop for one idle drive unit.

    The engine invokes this function once per idle unit, in unit-ID order. On
    the first invocation of each time step we plan one move for the entire
    idle fleet. Planning the moves together lets later units account for
    aisle and node slots reserved by earlier units.
    """
    global _actions_time, _actions, _last_time_seen, _pending_calls, _forecast
    global _seen_pods, _arrival_history

    _ensure_graph(state)

    # A lower time means GameEngine.reset() or a new test case in the same
    # interpreter. Graph-dependent caches are handled by _ensure_graph.
    if _last_time_seen is not None and state.current_time_step < _last_time_seen:
        _actions_time = None
        _actions = {}
        _forecast = {}
        _seen_pods = set()
        _arrival_history = {}

    if _actions_time != state.current_time_step or drive_unit_id not in _pending_calls:
        if _actions_time == state.current_time_step:
            # A second poll of the same unit at t=0 can also mean a new run.
            _forecast = {}
            _seen_pods = set()
            _arrival_history = {}
        _observe_arrivals(state)
        _actions = _plan_fleet_moves(state)
        _actions_time = state.current_time_step
        _pending_calls = {unit.id for unit in state.drive_units if not unit.in_transit}

    _pending_calls.discard(drive_unit_id)
    _last_time_seen = state.current_time_step
    return _actions.get(drive_unit_id)


def _ensure_graph(state: GraphState) -> None:
    """Build deterministic adjacency and all-pairs weighted distances."""
    global _graph_signature, _adjacency, _distances, _actions_time, _actions, _forecast
    global _seen_pods, _arrival_history

    signature = (
        tuple(sorted((n.id, n.node_type, -1 if n.capacity is None else n.capacity)
                     for n in state.nodes)),
        tuple((e.from_node, e.to_node, e.weight,
                      -1 if e.capacity is None else e.capacity,
                      e.bidirectional) for e in state.edges),
    )
    if signature == _graph_signature:
        return

    _graph_signature = signature
    _adjacency = {node.id: [] for node in state.nodes}
    for edge in state.edges:
        # GameEngine.get_edge() uses the first matching edge, so preserve that
        # behavior if an input happens to contain parallel endpoint pairs.
        forward = _adjacency.setdefault(edge.from_node, [])
        if not any(neighbor == edge.to_node for neighbor, _ in forward):
            forward.append((edge.to_node, edge))
        if edge.bidirectional:
            reverse = _adjacency.setdefault(edge.to_node, [])
            if not any(neighbor == edge.from_node for neighbor, _ in reverse):
                reverse.append((edge.from_node, edge))
    for neighbors in _adjacency.values():
        neighbors.sort(key=lambda pair: pair[0])

    _distances = {node.id: _dijkstra_static(node.id) for node in state.nodes}
    _actions_time = None
    _actions = {}
    _forecast = {}
    _seen_pods = set()
    _arrival_history = {}


def _dijkstra_static(start: int) -> Dict[int, float]:
    distances = {node_id: INF for node_id in _adjacency}
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, edge in _adjacency.get(node, []):
            candidate = distance + _travel_steps(edge.weight)
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _distance(start: int, end: int) -> float:
    return _distances.get(start, {}).get(end, INF)


def _travel_steps(weight: float) -> int:
    """The engine advances in whole ticks, so fractional weights round up."""
    return max(1, math.ceil(weight))


def _state_key(state):
    """Compare a forecast with the real state, including newly spawned pods."""
    return (
        tuple((u.id, u.current_node, u.capacity, u.in_transit,
               u.transit_destination, u.transit_remaining_time, tuple(u.carrying))
              for u in state.drive_units),
        tuple((p.id, p.current_node, p.carried_by, p.destination_station, p.entry_time)
              for p in state.active_pods),
    )


def _observe_arrivals(state):
    """Remember only arrivals actually observed through the public state."""
    for pod in state.active_pods:
        if pod.id in _seen_pods:
            continue
        _seen_pods.add(pod.id)
        source = pod.current_node
        if source is None:
            carrier = state.get_drive_unit(pod.carried_by)
            # A just-collected pod is still at its source before this poll.
            if carrier is not None and pod.entry_time == state.current_time_step:
                source = carrier.current_node
        if source is not None:
            waves = _arrival_history.setdefault(source, {})
            waves.setdefault(pod.entry_time, []).append(pod.deep_copy())


def _predict_arrivals(state):
    """A discounted estimate of one next wave, never a future-test lookup.

    Only consider a short batching wait for a partly loaded robot already at
    storage. Missed predictions expire at the observed cadence's next tick.
    """
    arrivals = {}
    now = state.current_time_step
    # Spare capacity is useful immediately when other pods are already waiting.
    # Speculative batching is reserved for gaps between workload waves.
    if any(p.carried_by is None for p in state.active_pods):
        return arrivals
    for unit in state.drive_units:
        if unit.in_transit or not unit.carrying or not unit.has_capacity:
            continue
        source = unit.current_node
        waves = _arrival_history.get(source, {})
        times = sorted(waves)
        if len(times) < 2:
            continue
        recent = times[-4:]
        gaps = [b - a for a, b in zip(recent, recent[1:])]
        period = gaps[-1]
        if period <= 0 or any(abs(gap - period) > 1 for gap in gaps):
            continue
        # With only two observations, require the same cadence at another
        # storage node; two isolated random arrivals are weak evidence.
        if len(times) == 2 and not any(
                other != source and all(t in history for t in times)
                for other, history in _arrival_history.items()):
            continue
        next_time = times[-1] + period
        if not 0 < next_time - now <= min(6, period / 2):
            continue
        key = (source, next_time)
        if key in arrivals:
            continue
        predicted = []
        # Use the observed destination mix, with uncertainty handled below.
        last_wave = waves[times[-1]]
        for index in range(min(len(last_wave), unit.capacity - len(unit.carrying))):
            pod = last_wave[index % len(last_wave)].deep_copy()
            pod.id = ("forecast", source, next_time, index)
            pod.current_node = source
            pod.carried_by = None
            pod.delivery_time = None
            pod.entry_time = next_time
            predicted.append(pod)
        arrivals[key] = predicted
    schedule = {}
    for (_, entry), pods in arrivals.items():
        schedule.setdefault(entry, []).extend(pods)
    return schedule


def _plan_fleet_moves(state):
    """Improve the dispatcher by simulating alternative jobs and first moves.

    Only the supplied state is used: future arrivals and test files are never
    inspected. Reuse the previous forecast while it matches observations, so
    a beneficial multi-step pickup detour is not abandoned halfway through.
    """
    global _forecast
    basic = _heuristic_moves(state)
    if (not state.active_pods or len(state.drive_units) > 8 or
            len(state.active_pods) > 32 or len(state.nodes) > 48):
        _forecast = {}
        return basic

    deadline = time.perf_counter() + 0.16
    arrivals = _predict_arrivals(state)
    cached = _forecast.get(state.current_time_step)
    incumbent = _forecast if cached and cached[0] == _state_key(state) else {}
    best = _rollout(state, {}, {}, incumbent, deadline, arrivals)
    if best is None:
        return basic
    if incumbent:
        alternative = _rollout(state, {}, {}, {}, deadline, arrivals)
        if alternative is not None and alternative[0] > best[0] + 1e-9:
            best = alternative

    waiting_sources = sorted({p.current_node for p in state.active_pods
                              if p.carried_by is None and p.current_node is not None})
    units = sorted((u for u in state.drive_units if not u.in_transit), key=lambda u: u.id)
    batch_holds = {u.id: t for u in units for t, wave in arrivals.items()
                   if any(p.current_node == u.current_node for p in wave)}
    if batch_holds:
        alternative = _rollout(state, {}, {}, best[1], deadline, arrivals, batch_holds)
        if alternative is not None and alternative[0] > best[0] + 1e-9:
            best = alternative
    trials = 0
    # Two passes can combine improvements for different robots while keeping
    # the search bounded. Each change is compared on full-fleet delivery score.
    for _ in range(2):
        improved = False
        for unit in units:
            missions = set()
            if unit.has_capacity:
                missions.update(waiting_sources)
            missions.update(state.get_pod(pid).destination_station for pid in unit.carrying)
            missions.discard(unit.current_node)
            candidates = [(target, False) for target in sorted(missions)
                          if math.isfinite(_distance(unit.current_node, target))]
            candidates += [(node, True) for node, _ in _adjacency.get(unit.current_node, [])]
            candidates.append((None, True))
            candidates += [(t, 2) for t, wave in arrivals.items()
                           if any(p.current_node == unit.current_node for p in wave)]

            for target, one_step in candidates:
                if trials >= 40 or time.perf_counter() >= deadline:
                    break
                plan = best[1]
                first = plan[state.current_time_step][1]
                if one_step == 1 and first.get(unit.id) == target:
                    continue
                trials += 1
                forced = {unit.id: target} if one_step == 1 else {}
                job = {} if one_step else {unit.id: target}
                holds = {unit.id: target} if one_step == 2 else {}
                alternative = _rollout(state, job, forced, plan, deadline, arrivals, holds)
                if alternative is not None and alternative[0] > best[0] + 1e-9:
                    best = alternative
                    improved = True
            if trials >= 40 or time.perf_counter() >= deadline:
                break
        if not improved or trials >= 40 or time.perf_counter() >= deadline:
            break

    _forecast = best[1]
    return _forecast[state.current_time_step][1]


def _rollout(state, missions, forced, existing, deadline, arrivals=None, holds=None):
    """A small forward model of the engine, starting just before routing.

    Pickups/deliveries run both after movement and at the next tick's start.
    Moves commit in unit-ID order; a destination slot is reserved on departure.
    """
    virtual = state.deep_copy()
    # Past deliveries are common to every candidate and need no simulation.
    virtual.delivered_pods = []
    missions = dict(missions)
    units = sorted(virtual.drive_units, key=lambda u: u.id)
    pods = {p.id: p for p in virtual.active_pods}
    plan = {}
    arrivals = arrivals or {}
    holds = holds or {}
    forecast_ids = {p.id for wave in arrivals.values() for p in wave}
    horizon = 96
    for offset in range(horizon):
        if offset % 8 == 0 and time.perf_counter() >= deadline:
            return None
        for unit in units:
            if not unit.in_transit and missions.get(unit.id) == unit.current_node:
                del missions[unit.id]
        key = _state_key(virtual)
        frame = existing.get(virtual.current_time_step)
        if not missions and frame and frame[0] == key:
            actions = dict(frame[1])
        else:
            actions = _heuristic_moves(virtual, missions)
        immediate = {uid: None for uid, until in holds.items() if virtual.current_time_step < until}
        if offset == 0:
            immediate.update(forced)
        if immediate:
            # Re-reserve later robots after an earlier robot changes its move.
            # In particular, yielding must let the next robot claim the dock.
            actions = _heuristic_moves(virtual, missions, immediate)
        executable = {}
        for unit in units:
            if unit.in_transit:
                continue
            hop = actions.get(unit.id)
            if hop is not None and _legal_now(virtual, unit, hop):
                edge = virtual.get_edge(unit.current_node, hop)
                unit.in_transit = True
                unit.transit_destination = hop
                unit.transit_remaining_time = edge.weight
                executable[unit.id] = hop
            else:
                executable[unit.id] = None
        plan[virtual.current_time_step] = (key, executable)
        for unit in units:
            if unit.in_transit:
                unit.transit_remaining_time -= 1
                if unit.transit_remaining_time <= 0:
                    unit.current_node = unit.transit_destination
                    unit.in_transit = False
                    unit.transit_destination = None
                    unit.transit_remaining_time = 0
        _resolve_pods(virtual, units, pods)
        virtual.current_time_step += 1
        for incoming in arrivals.get(virtual.current_time_step, []):
            pod = incoming.deep_copy()
            pods[pod.id] = pod
            virtual.active_pods.append(pod)
        if not virtual.active_pods and not any(t >= virtual.current_time_step for t in arrivals):
            break
        _resolve_pods(virtual, units, pods)
    score = sum((0.8 if p.id in forecast_ids else 1.0) * math.exp(-(p.delivery_time - p.entry_time) / 50.0)
                for p in virtual.delivered_pods)
    return score, plan


def _resolve_pods(state, units, pods):
    for unit in units:
        if unit.in_transit:
            continue
        for pid in list(unit.carrying):
            pod = pods[pid]
            if pod.destination_station == unit.current_node:
                unit.carrying.remove(pid)
                pod.carried_by = None
                pod.current_node = unit.current_node
                pod.delivery_time = state.current_time_step
                state.active_pods.remove(pod)
                state.delivered_pods.append(pod)
        if unit.has_capacity:
            waiting = sorted((p for p in state.active_pods
                              if p.carried_by is None and p.current_node == unit.current_node),
                             key=lambda p: (p.entry_time, p.id))
            for pod in waiting[:max(0, unit.capacity - len(unit.carrying))]:
                pod.carried_by = unit.id
                pod.current_node = None
                unit.carrying.append(pod.id)


def _heuristic_moves(state: GraphState, missions=None, forced=None) -> Dict[int, Optional[int]]:
    """Choose targets globally, then reserve this step's moves in ID order."""
    targets = _choose_fleet_targets(state)
    if missions:
        targets.update(missions)
    virtual = state.deep_copy()
    actions: Dict[int, Optional[int]] = {}

    for original in sorted(state.drive_units, key=lambda unit: unit.id):
        if original.in_transit:
            continue

        unit = virtual.get_drive_unit(original.id)
        target = targets.get(original.id)
        next_node = None
        if forced and original.id in forced:
            next_node = forced[original.id]
        elif unit is not None and target is not None and unit.current_node != target:
            next_node = _dynamic_next_hop(virtual, unit.id, target)

        if unit is not None and next_node is not None and _legal_now(virtual, unit, next_node):
            edge = virtual.get_edge(unit.current_node, next_node)
            unit.in_transit = True
            unit.transit_destination = next_node
            unit.transit_remaining_time = edge.weight
            actions[unit.id] = next_node
        elif unit is not None:
            actions[unit.id] = None

    return actions


def _choose_fleet_targets(state: GraphState) -> Dict[int, Optional[int]]:
    targets: Dict[int, Optional[int]] = {}

    # Loaded units are valuable and get an explicitly score-optimized delivery
    # order. Empty units are assigned jointly to the pods that are waiting.
    for unit in state.drive_units:
        if unit.carrying:
            targets[unit.id] = _best_delivery_target(unit, state)

    waiting = [pod for pod in state.active_pods
               if pod.carried_by is None and pod.current_node is not None]
    empty_units = [unit for unit in state.drive_units if not unit.carrying]
    pickup_targets = _assign_pickups(empty_units, waiting, state)
    targets.update(pickup_targets)

    # With no visible job, spread robots over storage nodes. Future pods are
    # not exposed in GraphState, so storage is the only useful staging signal.
    storage_nodes = sorted(node.id for node in state.nodes if node.node_type == "storage")
    staged_counts = {node_id: 0 for node_id in storage_nodes}
    for target in pickup_targets.values():
        if target in staged_counts:
            staged_counts[target] += 1

    for unit in sorted(empty_units, key=lambda item: item.id):
        if unit.id in targets:
            continue
        target = _best_storage_target(unit, storage_nodes, staged_counts)
        if target is None:
            target = _fallback_parking_target(unit, state)
        targets[unit.id] = target
        if target in staged_counts:
            staged_counts[target] += 1

    return targets


def _assign_pickups(units: Sequence, pods: Sequence, state: GraphState) -> Dict[int, int]:
    """Assign batches at source nodes to empty units.

    For small fleets we solve the discrete assignment exactly with dynamic
    programming. Larger hidden cases use a bounded greedy fallback so a
    routing call stays comfortably below the one-second limit.
    """
    if not units or not pods:
        return {}

    by_source: Dict[int, List] = {}
    for pod in pods:
        by_source.setdefault(pod.current_node, []).append(pod)
    sources = sorted(by_source)
    pod_lists = []
    for source in sources:
        ordered = sorted(by_source[source], key=lambda pod: (pod.entry_time, pod.id))
        pod_lists.append(ordered)

    ordered_units = sorted(units, key=lambda unit: unit.id)
    initial_remaining = tuple(len(items) for items in pod_lists)

    if (len(ordered_units) <= 5 and len(sources) <= 5 and
            max(unit.capacity for unit in ordered_units) <= 3):
        memo: Dict[Tuple[int, Tuple[int, ...]], Tuple[float, Tuple[Optional[int], ...]]] = {}

        def solve(index: int, remaining: Tuple[int, ...]):
            key = (index, remaining)
            if key in memo:
                return memo[key]
            if index == len(ordered_units):
                return 0.0, ()

            unit = ordered_units[index]
            best_score, suffix = solve(index + 1, remaining)
            best_choices: Tuple[Optional[int], ...] = (None,) + suffix

            for source_index, count in enumerate(remaining):
                if count <= 0:
                    continue
                take = min(unit.capacity, count)
                consumed_before = len(pod_lists[source_index]) - count
                batch = pod_lists[source_index][consumed_before:consumed_before + take]
                utility = _pickup_utility(unit, sources[source_index], batch, state)
                if utility <= 0:
                    continue
                next_remaining = list(remaining)
                next_remaining[source_index] -= take
                future_score, future_choices = solve(index + 1, tuple(next_remaining))
                score = utility + future_score
                choices = (source_index,) + future_choices
                if (score > best_score + 1e-12 or
                        (abs(score - best_score) <= 1e-12 and
                         _choice_key(choices, sources) < _choice_key(best_choices, sources))):
                    best_score = score
                    best_choices = choices

            memo[key] = best_score, best_choices
            return memo[key]

        _, choices = solve(0, initial_remaining)
        return {
            unit.id: sources[source_index]
            for unit, source_index in zip(ordered_units, choices)
            if source_index is not None
        }

    # Greedy fallback for unusually large fleets. Allocate the best remaining
    # unit/source batch at each iteration and consume the capacity it claims.
    remaining_units = list(ordered_units)
    remaining = list(initial_remaining)
    assignments: Dict[int, int] = {}
    while remaining_units and any(remaining):
        best = None
        for unit in remaining_units:
            for source_index, count in enumerate(remaining):
                if count <= 0:
                    continue
                take = min(unit.capacity, count)
                offset = len(pod_lists[source_index]) - count
                batch = pod_lists[source_index][offset:offset + take]
                utility = _pickup_utility(unit, sources[source_index], batch, state)
                candidate = (utility, -unit.id, -sources[source_index], unit, source_index, take)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 0:
            break
        _, _, _, unit, source_index, take = best
        assignments[unit.id] = sources[source_index]
        remaining[source_index] -= take
        remaining_units.remove(unit)
    return assignments


def _choice_key(choices: Sequence[Optional[int]], sources: Sequence[int]) -> Tuple[int, ...]:
    """Stable tie-breaker: real assignments first, then lower source IDs."""
    sentinel = (max(sources) + 1) if sources else 0
    return tuple(sentinel if choice is None else sources[choice] for choice in choices)


def _pickup_utility(unit, source: int, pods: Sequence, state: GraphState) -> float:
    eta = _unit_eta_to(unit, source)
    if not math.isfinite(eta):
        return 0.0
    score, _ = _best_order_score(source, pods, state.current_time_step + eta)
    return score


def _unit_eta_to(unit, target: int) -> float:
    if unit.in_transit:
        onward = _distance(unit.transit_destination, target)
        return unit.transit_remaining_time + onward
    return _distance(unit.current_node, target)


def _best_delivery_target(unit, state: GraphState) -> Optional[int]:
    pods = [state.get_pod(pod_id) for pod_id in unit.carrying]
    pods = [pod for pod in pods if pod is not None]
    if not pods:
        return None

    start = unit.transit_destination if unit.in_transit else unit.current_node
    _, order = _best_order_score(start, pods, state.current_time_step)
    return order[0] if order else None


def _best_order_score(start: int, pods: Sequence, start_time: float) -> Tuple[float, Tuple[int, ...]]:
    """Return predicted score and best order of the distinct destinations."""
    by_destination: Dict[int, List] = {}
    for pod in pods:
        by_destination.setdefault(pod.destination_station, []).append(pod)
    destinations = sorted(by_destination)
    if not destinations:
        return 0.0, ()

    if len(destinations) <= 5:
        candidates: Iterable[Tuple[int, ...]] = itertools.permutations(destinations)
    else:
        # Capacities are normally small. Keep pathological hidden inputs fast.
        remaining = set(destinations)
        greedy = []
        current = start
        while remaining:
            destination = min(remaining, key=lambda node: (_distance(current, node), node))
            greedy.append(destination)
            remaining.remove(destination)
            current = destination
        candidates = (tuple(greedy),)

    best_score = -1.0
    best_distance = INF
    best_order: Tuple[int, ...] = ()
    for order in candidates:
        current = start
        traveled = 0.0
        score = 0.0
        reachable = True
        for destination in order:
            leg = _distance(current, destination)
            if not math.isfinite(leg):
                reachable = False
                break
            traveled += leg
            delivery_time = start_time + max(0.0, traveled - 1.0)
            for pod in by_destination[destination]:
                duration = max(0.0, delivery_time - pod.entry_time)
                score += 100.0 * math.exp(-duration / 50.0)
            current = destination
        if not reachable:
            continue
        if (score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and
                 (traveled, order) < (best_distance, best_order))):
            best_score = score
            best_distance = traveled
            best_order = tuple(order)
    return max(0.0, best_score), best_order


def _best_storage_target(unit, storage_nodes: Sequence[int], staged_counts: Dict[int, int]) -> Optional[int]:
    if not storage_nodes:
        return None
    start = unit.transit_destination if unit.in_transit else unit.current_node
    reachable = [node for node in storage_nodes if math.isfinite(_distance(start, node))]
    if not reachable:
        return None

    # Do not make a robot leave a storage location merely to improve balancing.
    if not unit.in_transit and unit.current_node in reachable:
        return unit.current_node
    return min(reachable, key=lambda node: (
        _distance(start, node) + 3.0 * staged_counts.get(node, 0),
        _distance(start, node),
        node,
    ))


def _fallback_parking_target(unit, state: GraphState) -> Optional[int]:
    """Clear a station even in a malformed graph with no reachable storage."""
    if unit.in_transit:
        return None
    current = state.get_node(unit.current_node)
    if current is None or current.node_type != "station":
        return None
    candidates = _adjacency.get(unit.current_node, [])
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (
        state.get_node(pair[0]) is None or state.get_node(pair[0]).node_type == "station",
        _travel_steps(pair[1].weight),
        pair[0],
    ))[0]


def _dynamic_next_hop(state: GraphState, unit_id: int, target: int) -> Optional[int]:
    unit = state.get_drive_unit(unit_id)
    if unit is None or unit.in_transit or unit.current_node == target:
        return None

    start = unit.current_node
    distances = {node_id: INF for node_id in _adjacency}
    first_hop: Dict[int, int] = {}
    distances[start] = 0.0
    queue = [(0.0, start)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            break
        for neighbor, edge in _adjacency.get(node, []):
            cost = float(_travel_steps(edge.weight))
            cost += _edge_delay(state, edge)
            cost += _node_delay(state, neighbor)
            neighbor_model = state.get_node(neighbor)
            if neighbor_model is not None and neighbor_model.node_type == "station" and neighbor != target:
                cost += 0.5

            candidate = distance + cost
            proposed_first = neighbor if node == start else first_hop[node]
            old_first = first_hop.get(neighbor, 10 ** 18)
            if (candidate < distances[neighbor] - 1e-12 or
                    (abs(candidate - distances[neighbor]) <= 1e-12 and proposed_first < old_first)):
                distances[neighbor] = candidate
                first_hop[neighbor] = proposed_first
                heapq.heappush(queue, (candidate, neighbor))

    hop = first_hop.get(target)
    if hop is None:
        return None
    # If waiting for the preferred route is cheaper than a detour, Dijkstra may
    # intentionally select a currently full first hop. Waiting is then the
    # correct executable action for this step.
    return hop if _legal_now(state, unit, hop) else None


def _edge_delay(state: GraphState, edge) -> float:
    if edge.capacity is None:
        return 0.0
    occupants = [
        unit.transit_remaining_time
        for unit in state.drive_units
        if unit.in_transit and edge.connects(unit.current_node, unit.transit_destination)
    ]
    if len(occupants) >= edge.capacity:
        # The next slot opens when the earliest current traversal completes.
        return max(0.0, math.ceil(min(occupants)))
    # A slight utilization cost breaks equal-length ties away from busy aisles.
    return 0.15 * len(occupants) / edge.capacity


def _node_delay(state: GraphState, node_id: int) -> float:
    node = state.get_node(node_id)
    if node is None or node.capacity is None:
        return 0.0
    if state.node_occupancy(node_id) < node.capacity:
        return 0.0

    inbound = [unit.transit_remaining_time for unit in state.drive_units
               if unit.in_transit and unit.transit_destination == node_id]
    if inbound:
        return max(1.0, math.ceil(min(inbound)))
    # A parked occupant gets a routing opportunity every step. This penalty
    # encourages useful detours without treating the node as permanently shut.
    return 2.0


def _legal_now(state: GraphState, unit, next_node: int) -> bool:
    if unit.in_transit or next_node == unit.current_node:
        return False
    edge = state.get_edge(unit.current_node, next_node)
    if edge is None:
        return False
    if edge.capacity is not None:
        if state.edge_occupancy(unit.current_node, next_node) >= edge.capacity:
            return False
    node = state.get_node(next_node)
    if node is None:
        return False
    if node.capacity is not None and state.node_occupancy(next_node) >= node.capacity:
        return False
    return True
