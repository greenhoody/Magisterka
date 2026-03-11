from pprint import pprint

from botbowl import OutcomeType, Game
import botbowl.core.procedure as procedure
from examples.scripted_bot_example import MyScriptedBot


class A2C_Reward:
    # --- Reward function ---
    rewards_own = {
        OutcomeType.TOUCHDOWN: 2,
        OutcomeType.SUCCESSFUL_CATCH: 0.1,
        OutcomeType.INTERCEPTION: 0.2,
        OutcomeType.SUCCESSFUL_PICKUP: 0.1,
        OutcomeType.FUMBLE: -0.1,
        OutcomeType.KNOCKED_DOWN: -0.1,
        OutcomeType.KNOCKED_OUT: -0.2,
        OutcomeType.CASUALTY: -0.5
    }
    rewards_opp = {
        OutcomeType.TOUCHDOWN: -2,
        OutcomeType.SUCCESSFUL_CATCH: -0.1,
        OutcomeType.INTERCEPTION: -0.2,
        OutcomeType.SUCCESSFUL_PICKUP: -0.1,
        OutcomeType.FUMBLE: 0.1,
        OutcomeType.KNOCKED_DOWN: 0.1,
        OutcomeType.KNOCKED_OUT: 0.2,
        OutcomeType.CASUALTY: 0.5
    }
    # Old step-wise shaping (used when use_turn_end_rewards=False)
    ball_progression_reward = 0.005
    # New turn-end shaping
    ball_position_weight = 1.0
    win_reward = 10.0
    loss_reward = -10.0
    draw_reward = 0.0

    def __init__(self, use_turn_end_rewards: bool = True):
        self.use_turn_end_rewards = use_turn_end_rewards
        self.last_report_idx = 0
        self.last_ball_x = None
        self.last_ball_team = None
        self.pending_events = []

    def __call__(self, game: Game):
        if not self.use_turn_end_rewards:
            return self._stepwise_reward(game)

        if len(game.state.reports) < self.last_report_idx:
            self.last_report_idx = 0
            self.pending_events = []

        r = 0.0
        reports = game.state.reports[self.last_report_idx:]
        self.last_report_idx = len(game.state.reports)

        for outcome in reports:
            self.pending_events.append(outcome)
            if outcome.outcome_type in (
                OutcomeType.END_OF_TURN,
                OutcomeType.END_OF_BLITZ,
                OutcomeType.END_OF_QUICK_SNAP,
            ):
                turn_team = outcome.team or game.active_team
                r = self._score_events(game, turn_team)
                self.pending_events = []
                break

        return r

    def _stepwise_reward(self, game: Game) -> float:
        if len(game.state.reports) < self.last_report_idx:
            self.last_report_idx = 0

        r = 0.0
        own_team = game.active_team
        opp_team = game.get_opp_team(own_team)

        for outcome in game.state.reports[self.last_report_idx:]:
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team
            if team == own_team and outcome.outcome_type in A2C_Reward.rewards_own:
                r += A2C_Reward.rewards_own[outcome.outcome_type]
            if team == opp_team and outcome.outcome_type in A2C_Reward.rewards_opp:
                r += A2C_Reward.rewards_opp[outcome.outcome_type]

        self.last_report_idx = len(game.state.reports)

        ball_carrier = game.get_ball_carrier()
        if ball_carrier is not None:
            if self.last_ball_team is own_team and ball_carrier.team is own_team:
                ball_progress = self.last_ball_x - ball_carrier.position.x
                if own_team is game.state.away_team:
                    ball_progress *= -1  # End zone at max x coordinate
                r += A2C_Reward.ball_progression_reward * ball_progress

            self.last_ball_team = ball_carrier.team
            self.last_ball_x = ball_carrier.position.x
        else:
            self.last_ball_team = None
            self.last_ball_x = None

        return r

    def _score_events(self, game: Game, turn_team) -> float:
        r = 0.0
        opp_team = game.get_opp_team(turn_team)

        for outcome in self.pending_events:
            team = None
            if outcome.player is not None:
                team = outcome.player.team
            elif outcome.team is not None:
                team = outcome.team

            if team == turn_team and outcome.outcome_type in A2C_Reward.rewards_own:
                r += A2C_Reward.rewards_own[outcome.outcome_type]
            if team == opp_team and outcome.outcome_type in A2C_Reward.rewards_opp:
                r += A2C_Reward.rewards_opp[outcome.outcome_type]

            r += self._ball_position_value(game, turn_team, outcome)

            if outcome.outcome_type == OutcomeType.END_OF_GAME_WINNER:
                r += A2C_Reward.win_reward if outcome.team == turn_team else A2C_Reward.loss_reward
            elif outcome.outcome_type == OutcomeType.END_OF_GAME_DRAW:
                r += A2C_Reward.draw_reward

        return r

    def _ball_position_value(self, game: Game, turn_team, outcome) -> float:
        pos = outcome.position or game.get_ball_position()
        if pos is None:
            return 0.0
        width = max(1, game.arena.width - 1)
        x = pos.x / width
        if turn_team is game.state.away_team:
            x = 1.0 - x
        return A2C_Reward.ball_position_weight * x


def a2c_scripted_actions(game: Game):
    proc_type = type(game.get_procedure())
    if proc_type is procedure.Block:
        # noinspection PyTypeChecker
        return MyScriptedBot.block(self=None, game=game)

    return None
