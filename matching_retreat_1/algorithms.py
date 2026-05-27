# matching_live/algorithms.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import numpy as np

from scipy.stats import binomtest


Choice = str  # "L" | "R" | "M"


@dataclass
class NGramBinaryTable:
    """
    For MatchingPennies1: base-2 encoding of last n choices (ignoring "M").
    Stores per-context:
      - left_counts[idx]
      - totals[idx]
      - pvalues[idx] (binomtest(left_counts, totals, p=0.5))
    """
    n: int
    left_counts: np.ndarray
    totals: np.ndarray
    pvalues: np.ndarray

    @classmethod
    def initialize(cls, n: int) -> "NGramBinaryTable":
        size = 2 ** n if n > 0 else 1
        return cls(
            n=n,
            left_counts=np.zeros(size, dtype=np.int64),
            totals=np.zeros(size, dtype=np.int64),
            pvalues=np.full(size, np.nan, dtype=float),
        )

    def update(self, idx: int, last_choice: Choice) -> None:
        is_left = 1 if last_choice == "L" else 0
        self.left_counts[idx] += is_left
        self.totals[idx] += 1
        self.pvalues[idx] = binomtest(int(self.left_counts[idx]), int(self.totals[idx]), 0.5).pvalue

    def row(self, idx: int) -> Tuple[int, int, float]:
        return int(self.left_counts[idx]), int(self.totals[idx]), float(self.pvalues[idx])


class MatchingPennies1:
    """
    Python port of MATLAB MatchingPennies1.

    - Maintains n-gram tables for trial_back = 0..N over *choices only*.
    - Ignores "M" choices when building the history index (like MATLAB).
    - Uses binomial test p-values; on sample picks the row with minimum p-value.

    NOTE: MATLAB uses:
        trial_types((left/total > 0.5) + 1) with ["L","R"]
      => if p_left > 0.5 choose "R" else "L"
    We preserve that behavior for fidelity.
    """

    TRIAL_TYPES: Tuple[str, str] = ("L", "R")

    def __init__(self, N: int, alpha: float, rng: Optional[np.random.Generator] = None):
        self.trials_back = int(N)
        self.alpha = float(alpha)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.choice_history: List[Choice] = []
        self.reward_history: List[int] = []
        self.tt_history: List[Choice] = []

        # ngram[trial_back] for trial_back=0..N
        self.ngram: List[NGramBinaryTable] = [
            NGramBinaryTable.initialize(n=trial_back) for trial_back in range(0, self.trials_back + 1)
        ]

    def update(self, last_choice: Choice, last_reward: int = 1) -> None:
        if last_choice not in ("L", "R", "M"):
            raise ValueError(f"last_choice must be 'L', 'R', or 'M', got {last_choice!r}")

        # MATLAB: if ismember(last_choice, trial_types) then update ngrams
        if last_choice in self.TRIAL_TYPES:
            for trial_back in range(0, self.trials_back + 1):
                self._update_ngram(trial_back=trial_back, last_choice=last_choice)

        self.choice_history.append(last_choice)
        self.reward_history.append(int(last_reward))

    def _update_ngram(self, trial_back: int, last_choice: Choice) -> None:
        # MATLAB: if trial_back < length(choice_history)
        # (length(choice_history) is BEFORE appending current trial)
        if trial_back < len(self.choice_history):
            idx = self._history_to_index(trial_back)
            self.ngram[trial_back].update(idx=idx, last_choice=last_choice)

    def _history_to_index(self, trial_back: int) -> int:
        # MATLAB:
        # if trial_back == 0: idx=1
        # else:
        #   choice_history = choice_history; remove "M"
        #   last_n = (last trial_back choices == "L") -> booleans
        #   idx = dot(2.^(0:numel-1), last_n) + 1
        if trial_back == 0:
            return 0  # 0-based

        ch = [c for c in self.choice_history if c != "M"]
        if len(ch) < trial_back:
            # MATLAB would error if asked too early; we only call when trial_back < len(choice_history)
            # but "M" removal can reduce it further. In that case, just map to 0 (safest).
            return 0

        last = ch[-trial_back:]  # oldest..newest slice
        bits = [1 if c == "L" else 0 for c in last]
        idx0 = int(sum(bit * (2 ** i) for i, bit in enumerate(bits)))
        return idx0

    def relevant_rows(self) -> List[Tuple[int, int, float]]:
        rows: List[Tuple[int, int, float]] = []
        for trial_back in range(0, self.trials_back + 1):
            # only include if MATLAB would (trial_back < length(choice_history))
            if trial_back < len(self.choice_history):
                idx = self._history_to_index(trial_back)
                rows.append(self.ngram[trial_back].row(idx))
        return rows

    def sample(self) -> Choice:
        if self.trials_back < len(self.choice_history):
            rows = self.relevant_rows()
            if not rows:
                choice = self._random_choice()
            else:
                pvals = np.array([r[2] for r in rows], dtype=float)
                pvals = np.where(np.isnan(pvals), np.inf, pvals)
                best_i = int(np.argmin(pvals))
                left, total, pval = rows[best_i]

                if not np.isfinite(pval) or pval > self.alpha or total <= 0:
                    choice = self._random_choice()
                else:
                    p_left = left / total
                    # preserve MATLAB mapping
                    choice = "R" if (p_left > 0.5) else "L"
        else:
            choice = self._random_choice()

        self.tt_history.append(choice)
        return choice

    def _random_choice(self) -> Choice:
        return self.TRIAL_TYPES[int(self.rng.integers(0, 2))]


@dataclass
class NGramChoiceRewardTable:
    """
    For MatchingPennies2 core: base-4 encoding of last n tokens among:
      "L 0","R 0","L 1","R 1" (ignores "M 0" in indexing).
    """
    n: int
    left_counts: np.ndarray
    totals: np.ndarray
    pvalues: np.ndarray

    @classmethod
    def initialize(cls, n: int) -> "NGramChoiceRewardTable":
        size = 4 ** n
        return cls(
            n=n,
            left_counts=np.zeros(size, dtype=np.int64),
            totals=np.zeros(size, dtype=np.int64),
            pvalues=np.full(size, np.nan, dtype=float),
        )

    def update(self, idx: int, last_choice: Choice) -> None:
        is_left = 1 if last_choice == "L" else 0
        self.left_counts[idx] += is_left
        self.totals[idx] += 1
        self.pvalues[idx] = binomtest(int(self.left_counts[idx]), int(self.totals[idx]), 0.5).pvalue

    def row(self, idx: int) -> Tuple[int, int, float]:
        return int(self.left_counts[idx]), int(self.totals[idx]), float(self.pvalues[idx])


class MatchingPennies2:
    """
    Python port of MATLAB MatchingPennies2.
    Includes an embedded MatchingPennies1 (mp1) and pools rows, choosing min p-value.

    invert_prediction:
      - MATLAB returns a "choice" derived from the selected row.
      - If you're using this as an *opponent* against the human, you probably want
        to "exploit" (pick the opposite of predicted human choice). That corresponds
        to invert_prediction=True in many setups.
    """

    TRIAL_TYPES: Tuple[str, str] = ("L", "R")
    MISS_TOKEN: str = "M 0"
    TOKEN_TO_DIGIT = {"L 0": 0, "R 0": 1, "L 1": 2, "R 1": 3}

    def __init__(
        self,
        N: int,
        alpha: float,
        *,
        invert_prediction: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.trials_back = int(N)
        self.alpha = float(alpha)
        self.invert_prediction = bool(invert_prediction)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.choice_history: List[Choice] = []
        self.reward_history: List[int] = []
        self.choice_reward_combination: List[str] = []
        self.tt_history: List[Choice] = []

        # MP2 ngram tables for n=1..N (index 0 unused)
        self.ngram: List[Optional[NGramChoiceRewardTable]] = [None]
        for n in range(1, self.trials_back + 1):
            self.ngram.append(NGramChoiceRewardTable.initialize(n))

        # Embedded MP1 (choices-only)
        self.mp1 = MatchingPennies1(N=self.trials_back, alpha=self.alpha, rng=self.rng)

    def update(self, last_choice: Choice, last_reward: int) -> None:
        if last_choice not in ("L", "R", "M"):
            raise ValueError(f"last_choice must be 'L', 'R', or 'M', got {last_choice!r}")
        last_reward_int = int(last_reward)
        if last_reward_int not in (0, 1):
            raise ValueError(f"last_reward must be 0/1, got {last_reward!r}")

        # Update embedded MP1 regardless (it ignores reward)
        self.mp1.update(last_choice, last_reward_int)

        last_trial = f"{last_choice} {last_reward_int}"

        if last_trial != self.MISS_TOKEN:
            for trial_back in range(1, self.trials_back + 1):
                self._update_ngram(trial_back=trial_back, last_trial=last_trial)

        self.choice_history.append(last_choice)
        self.reward_history.append(last_reward_int)
        self.choice_reward_combination.append(last_trial)

    def _update_ngram(self, trial_back: int, last_trial: str) -> None:
        if trial_back < len(self.choice_history):
            idx = self._history_to_index(trial_back)
            table = self.ngram[trial_back]
            assert table is not None
            table.update(idx=idx, last_choice=last_trial[0])

    def _history_to_index(self, trial_back: int) -> int:
        trial_combination = [t for t in self.choice_reward_combination if t != self.MISS_TOKEN]
        if len(trial_combination) < trial_back:
            return 0
        recent = trial_combination[-trial_back:]  # oldest..newest
        digits = [self.TOKEN_TO_DIGIT[t] for t in recent]
        idx0 = int(sum(d * (4 ** i) for i, d in enumerate(digits)))
        return idx0

    def relevant_rows(self) -> List[Tuple[int, int, float]]:
        rows: List[Tuple[int, int, float]] = []

        # MP2 rows for n=1..N
        for n in range(1, self.trials_back + 1):
            if n < len(self.choice_history):
                idx = self._history_to_index(n)
                table = self.ngram[n]
                assert table is not None
                rows.append(table.row(idx))

        # Add MP1 rows
        rows.extend(self.mp1.relevant_rows())
        return rows

    def sample(self) -> Choice:
        if self.trials_back < len(self.choice_history):
            rows = self.relevant_rows()
            if not rows:
                choice = self._random_choice()
            else:
                pvals = np.array([r[2] for r in rows], dtype=float)
                pvals = np.where(np.isnan(pvals), np.inf, pvals)
                best_i = int(np.argmin(pvals))
                left, total, pval = rows[best_i]

                if not np.isfinite(pval) or pval > self.alpha or total <= 0:
                    choice = self._random_choice()
                else:
                    p_left = left / total
                    predicted = "R" if (p_left > 0.5) else "L"  # preserve MATLAB mapping
                    if self.invert_prediction:
                        choice = "L" if predicted == "R" else "R"
                    else:
                        choice = predicted
        else:
            choice = self._random_choice()

        self.tt_history.append(choice)
        return choice

    def _random_choice(self) -> Choice:
        return self.TRIAL_TYPES[int(self.rng.integers(0, 2))]

    def to_dict(self) -> dict:
        """
        Lightweight state summary (safe to JSON-dump).
        """
        d = {
            "trials_back": self.trials_back,
            "alpha": self.alpha,
            "n_trials_seen": len(self.choice_history),
        }
        # Optional: add current best pval / estimate
        if self.trials_back < len(self.choice_history):
            rows = self.relevant_rows()
            if rows:
                pvals = np.array([r[2] for r in rows], dtype=float)
                pvals = np.where(np.isnan(pvals), np.inf, pvals)
                bi = int(np.argmin(pvals))
                left, total, pval = rows[bi]
                d.update(
                    best_row_index=bi,
                    best_pvalue=float(pval) if np.isfinite(pval) else None,
                    best_p_left=(left / total) if total > 0 else None,
                )
        return d
    
    def _safe_history_index(self, n: int):
        """Return (idx, context_str) for the current n-back context, or (None, None) if unavailable."""
        if n >= len(self.choice_history):
            return None, None
        idx = self._history_to_index(n)
        table = self.ngram[n]
        ctx = table.combos[idx] if (table is not None and hasattr(table, "combos")) else None

        return idx, ctx


    def debug_relevant(self):
        """
        Return detailed per-n-back info for the current state:
        list of dicts with n, idx, context, left, total, pval, p_left
        """
        out = []
        for n in range(1, self.trials_back + 1):
            idx, ctx = self._safe_history_index(n)
            if idx is None:
                continue
            table = self.ngram[n]
            left, total, pval = table.relevant_row(idx)
            p_left = (left / total) if total > 0 else np.nan
            out.append(
                dict(n=n, idx=idx, context=ctx, left=left, total=total, pval=pval, p_left=p_left)
            )
        return out


    def debug_decision(self):
        """
        Compute what sample() would base its decision on (without sampling randomness for the fallback case).
        Returns dict with best_n, best_pval, best_p_left, and 'mode' in {'random','rule'}.
        """
        rows = self.debug_relevant()
        if not rows:
            return dict(mode="random", reason="not_enough_history")

        # choose min pval (ignore NaNs)
        pvals = np.array([r["pval"] for r in rows], dtype=float)
        pvals = np.where(np.isnan(pvals), np.inf, pvals)
        best_i = int(np.argmin(pvals))
        best = rows[best_i]

        if not np.isfinite(best["pval"]) or best["pval"] > self.alpha or best["total"] <= 0:
            return dict(mode="random", reason="pval>alpha_or_unseen", best=best)

        # MATLAB rule: predicted = "R" if p_left > 0.5 else "L"
        predicted = "R" if (best["p_left"] > 0.5) else "L"
        choice_if_no_invert = predicted
        choice_if_invert = ("L" if predicted == "R" else "R")

        return dict(
            mode="rule",
            best=best,
            predicted=predicted,
            choice_if_invert=choice_if_invert,
            choice_if_no_invert=choice_if_no_invert,
        )
    

# Two-armed bandit task 
@dataclass
class BlockFlipperWithExtension:
    """
    Non-adaptive 2AB environment:
      - reward probs are (p_high, p_low) assigned to (L,R) depending on block
      - block length ~ Poisson(lambda_)
      - when block_flip < extend_block, extend block if recent high-choice rate < threshold

    Mirrors the MATLAB behavior closely.
    """
    p_high: float = 0.9
    p_low: float = 0.0
    lambda_: float = 25.0

    extend_block: int = 5
    block_extend_threshold: float = 0.2  # extend if chose-high proportion is below this

    rng: Optional[np.random.Generator] = None

    # state
    high_side: str = "L"          # which side is currently high-prob ("L" or "R")
    block_length: int = 0
    block_flip: int = 0
    n_trials: int = 0

    # store whether the human chose the high side each trial (1/0); used for extension rule
    chose_high_hist: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

        # MATLAB: obj.block = binornd(1,0.5); new_block flips it
        # We'll start by randomizing high_side, then immediately "flip" once, similar effect.
        self.high_side = "L" if self.rng.binomial(1, 0.5) == 1 else "R"
        self._new_block()  # flip once like MATLAB new_block()

    def _draw_block_length(self) -> int:
        L = int(self.rng.poisson(self.lambda_))
        return max(1, L)  # avoid zero-length blocks

    def _new_block(self) -> None:
        # flip high side
        self.high_side = "R" if self.high_side == "L" else "L"
        self.block_length = self._draw_block_length()
        self.block_flip = self.block_length

    def reward_prob(self, choice: str) -> float:
        if choice not in ("L", "R"):
            raise ValueError("choice must be 'L' or 'R'")
        return self.p_high if choice == self.high_side else self.p_low

    def trial(self, human_choice: str) -> int:
        """
        Execute one trial given human choice.
        Returns reward_bin in {0,1}.
        Handles block extension + countdown update.
        """
        if human_choice not in ("L", "R"):
            raise ValueError("human_choice must be 'L' or 'R'")

        # deliver probabilistic reward based on current block
        p = self.reward_prob(human_choice)
        reward_bin = int(self.rng.binomial(1, p))

        # track whether chose high side
        chose_high = 1 if human_choice == self.high_side else 0
        self.chose_high_hist.append(chose_high)
        self.n_trials += 1

        # --- extension logic (mirrors MATLAB intent) ---
        # In MATLAB, they check when block_flip < extend_block (last few trials),
        # then compute a mean over (block_length - extend_block) ... end.
        # We'll approximate that by looking at the CURRENT block's history tail of length extend_block,
        # which is typically what you want behaviorally.
        if self.block_flip < self.extend_block:
            recent = self.chose_high_hist[-self.extend_block:] if len(self.chose_high_hist) >= self.extend_block else self.chose_high_hist
            if len(recent) > 0 and (sum(recent) / len(recent)) < self.block_extend_threshold:
                self.block_flip += self.extend_block
                self.block_length += self.extend_block

        # decrement / flip block for next trial
        if self.block_flip <= 1:
            self._new_block()
        else:
            self.block_flip -= 1

        return reward_bin

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "BlockFlipperWithExtension",
            "p_high": self.p_high,
            "p_low": self.p_low,
            "lambda": self.lambda_,
            "extend_block": self.extend_block,
            "block_extend_threshold": self.block_extend_threshold,
            "high_side": self.high_side,
            "block_length": self.block_length,
            "block_flip": self.block_flip,
            "n_trials": self.n_trials,
        }