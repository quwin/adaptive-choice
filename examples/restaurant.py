"""An adaptive restaurant-choice simulation.

This example deliberately keeps every domain object application-owned.  The
world decides which restaurants are open, the observer exposes imperfect wait
estimates, the model assigns utilities, and the updater learns a separate
affinity for each visited restaurant.

Run it from an installed source checkout with::

    python examples/restaurant.py --seed 7 --steps 8
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from adaptive_choice import DecisionSystem, SoftmaxSampler, StepResult


@dataclass(frozen=True, slots=True)
class Restaurant:
    name: str
    cuisine: str
    price: float
    advertised_rating: float
    actual_quality: float
    base_wait: int


@dataclass(frozen=True, slots=True)
class VenueStatus:
    restaurant: Restaurant
    actual_wait: int


@dataclass(frozen=True, slots=True)
class RestaurantWorld:
    day: int
    open_venues: tuple[VenueStatus, ...]


@dataclass(frozen=True, slots=True)
class RestaurantView:
    name: str
    cuisine: str
    believed_price: float
    believed_rating: float
    believed_wait: int


@dataclass(frozen=True, slots=True)
class RestaurantObservation:
    day: int
    options: tuple[RestaurantView, ...]


@dataclass(frozen=True, slots=True)
class EatAt:
    restaurant: Restaurant


@dataclass(frozen=True, slots=True)
class DiningOutcome:
    restaurant_name: str
    cost: float
    wait: int
    satisfaction: float


@dataclass(frozen=True, slots=True)
class Diner:
    cuisine_preferences: Mapping[str, float]
    restaurant_affinities: Mapping[str, float]
    known_restaurants: frozenset[str]
    budget: float
    price_sensitivity: float
    wait_sensitivity: float
    learning_rate: float
    wait_estimate_bias: int = -3
    visits: tuple[str, ...] = ()


class RestaurantEnvironment:
    """Authoritative, deterministic restaurant availability and outcomes."""

    def __init__(self, restaurants: Sequence[Restaurant]) -> None:
        self._restaurants = tuple(restaurants)
        self._day = 0

    @property
    def state(self) -> RestaurantWorld:
        venues: list[VenueStatus] = []
        for index, restaurant in enumerate(self._restaurants):
            # Every fourth day all venues open; otherwise one rotates closed.
            if self._day % 4 != 3 and index == self._day % len(self._restaurants):
                continue
            wait_variation = (self._day * 7 + index * 3) % 11
            venues.append(
                VenueStatus(
                    restaurant=restaurant,
                    actual_wait=restaurant.base_wait + wait_variation,
                )
            )
        return RestaurantWorld(day=self._day, open_venues=tuple(venues))

    def legal_actions(self, agent: Diner) -> tuple[EatAt, ...]:
        return tuple(
            EatAt(status.restaurant)
            for status in self.state.open_venues
            if status.restaurant.name in agent.known_restaurants
            and status.restaurant.price <= agent.budget
        )

    def step(self, action: EatAt) -> DiningOutcome:
        open_by_name = {
            status.restaurant.name: status for status in self.state.open_venues
        }
        try:
            status = open_by_name[action.restaurant.name]
        except KeyError as error:
            raise ValueError(
                f"{action.restaurant.name!r} is not currently open"
            ) from error

        # The objective dining outcome is deterministic. Choice is the only
        # stochastic component in this example.
        satisfaction = max(
            0.0,
            min(1.0, status.restaurant.actual_quality - 0.004 * status.actual_wait),
        )
        outcome = DiningOutcome(
            restaurant_name=status.restaurant.name,
            cost=status.restaurant.price,
            wait=status.actual_wait,
            satisfaction=satisfaction,
        )
        self._day += 1
        return outcome


class DinerObserver:
    """Expose only known venues and the diner's biased wait estimates."""

    def observe(
        self, state: RestaurantWorld, agent: Diner
    ) -> RestaurantObservation:
        options = tuple(
            RestaurantView(
                name=status.restaurant.name,
                cuisine=status.restaurant.cuisine,
                believed_price=status.restaurant.price,
                believed_rating=status.restaurant.advertised_rating,
                believed_wait=max(0, status.actual_wait + agent.wait_estimate_bias),
            )
            for status in state.open_venues
            if status.restaurant.name in agent.known_restaurants
        )
        return RestaurantObservation(day=state.day, options=options)


class RestaurantUtilityModel:
    """A hand-written, candidate-local utility model."""

    def logits(
        self,
        observation: RestaurantObservation,
        agent: Diner,
        actions: Sequence[EatAt],
    ) -> tuple[float, ...]:
        views = {view.name: view for view in observation.options}
        scores: list[float] = []
        for action in actions:
            view = views[action.restaurant.name]
            cuisine_utility = agent.cuisine_preferences.get(view.cuisine, 0.0)
            learned_affinity = agent.restaurant_affinities.get(view.name, 0.0)
            scores.append(
                1.4 * cuisine_utility
                + 1.8 * view.believed_rating
                + learned_affinity
                - agent.price_sensitivity * view.believed_price
                - agent.wait_sensitivity * view.believed_wait
            )
        return tuple(scores)


class ExperienceUpdater:
    """Learn affinity from the gap between advertised and experienced quality."""

    def update(
        self,
        agent: Diner,
        observation: RestaurantObservation,
        action: EatAt,
        outcome: DiningOutcome,
    ) -> Diner:
        views = {view.name: view for view in observation.options}
        restaurant_name = action.restaurant.name
        prediction = views[restaurant_name].believed_rating
        prediction_error = outcome.satisfaction - prediction
        affinities = dict(agent.restaurant_affinities)
        affinities[restaurant_name] = affinities.get(restaurant_name, 0.0) + (
            agent.learning_rate * prediction_error
        )
        return replace(
            agent,
            restaurant_affinities=affinities,
            visits=agent.visits + (restaurant_name,),
        )


@dataclass(frozen=True, slots=True)
class SimulationRun:
    results: tuple[
        StepResult[RestaurantObservation, EatAt, DiningOutcome, Diner], ...
    ]
    final_agent: Diner


def build_restaurants() -> tuple[Restaurant, ...]:
    return (
        Restaurant("Saffron Bowl", "Indian", 18.0, 0.91, 0.95, 12),
        Restaurant("Harbor Tacos", "Mexican", 14.0, 0.86, 0.80, 8),
        Restaurant("Garden Table", "Vegetarian", 16.0, 0.88, 0.90, 10),
    )


def build_diner(restaurants: Sequence[Restaurant]) -> Diner:
    return Diner(
        cuisine_preferences={"Indian": 0.9, "Mexican": 0.7, "Vegetarian": 0.6},
        restaurant_affinities={},
        known_restaurants=frozenset(restaurant.name for restaurant in restaurants),
        budget=20.0,
        price_sensitivity=0.055,
        wait_sensitivity=0.025,
        learning_rate=0.6,
    )


def run_simulation(*, seed: int = 7, steps: int = 8) -> SimulationRun:
    """Run a reproducible trajectory and return every immutable step record."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    restaurants = build_restaurants()
    environment = RestaurantEnvironment(restaurants)
    agent = build_diner(restaurants)
    system = DecisionSystem(
        observer=DinerObserver(),
        choice_model=RestaurantUtilityModel(),
        sampler=SoftmaxSampler(temperature=0.75),
        updater=ExperienceUpdater(),
    )
    rng = random.Random(seed)
    results: list[
        StepResult[RestaurantObservation, EatAt, DiningOutcome, Diner]
    ] = []

    for _ in range(steps):
        result = system.step(environment=environment, agent=agent, rng=rng)
        results.append(result)
        agent = result.agent

    return SimulationRun(results=tuple(results), final_agent=agent)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="choice RNG seed")
    parser.add_argument(
        "--steps",
        type=_non_negative_int,
        default=8,
        help="number of decisions to simulate",
    )
    args = parser.parse_args(argv)
    run = run_simulation(seed=args.seed, steps=args.steps)

    print(f"Adaptive restaurant choices (seed={args.seed}, steps={args.steps})")
    for result in run.results:
        outcome = result.outcome
        selected_probability = result.choice.probabilities[result.choice.index]
        print(
            f"day {result.observation.day + 1:>2}: "
            f"{outcome.restaurant_name:<15} "
            f"p={selected_probability:.3f} "
            f"cost=${outcome.cost:.2f} "
            f"wait={outcome.wait:>2}m "
            f"satisfaction={outcome.satisfaction:.3f}"
        )

    print("Learned restaurant affinities:")
    for name in sorted(run.final_agent.restaurant_affinities):
        value = run.final_agent.restaurant_affinities[name]
        print(f"  {name:<15} {value:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
