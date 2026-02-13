from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

try:
    from scipy.stats import binomtest
except Exception as e:
    raise ImportError("This implementation requires SciPy (for scipy.stats.binomtest).") from e


Choice = str  # "L" | "R" | "M"


@dataclass
class NGramTable:
    """
    Stores counts for a fixed n (trial_back):
      - left_counts[idx]: number of times next choice was 'L' for this context
      - totals[idx]: number of observations for this context
      - pvalues[idx]: binomial test p-value for left_counts vs totals under p=0.5
    Indexing uses base-4 encoding of the last n (choice, reward) tokens.
    """
    n: int
    left_counts: np.ndarray  # shape (4**n,)
    totals: np.ndarray       # shape (4**n,)
    pvalues: np.ndarray      # shape (4**n,)
    combos: List[str]        # optional: human-readable context labels, length 4**n

    @classmethod
    def initialize(cls, n: int) -> "NGramTable":
        size = 4 ** n
        left_counts = np.zeros(size, dtype=np.int64)
        totals = np.zeros(size, dtype=np.int64)
        pvalues = np.full(size, np.nan, dtype=float)

        # Recreate the MATLAB "all_combinations" strings ("L 0", "R 0", "L 1", "R 1")
        token = np.array(["L 0", "R 0", "L 1", "R 1"], dtype=object)

        # combos[idx] should correspond to the base-4 digits for that idx
        # where digit 0 is the most recent trial in their encoding (4^(0:numel-1)).
        # We'll build strings like: "L 0 | R 1 | L 1" (most recent first)
        combos: List[str] = []
        for idx in range(size):
            digits = []
            x = idx
            for _ in range(n):
                digits.append(x % 4)
                x //= 4
            # digits are least-significant first => aligns with 4^0 being "most recent"
            combos.append(" | ".join(token[d] for d in digits))

        return cls(n=n, left_counts=left_counts, totals=totals, pvalues=pvalues, combos=combos)

    def update(self, idx: int, last_choice: Choice) -> None:
        # MATLAB increments:
        #   [extract(last_trial,1)=="L", 1, 0]
        is_left = 1 if last_choice == "L" else 0
        self.left_counts[idx] += is_left
        self.totals[idx] += 1
        self.pvalues[idx] = binomtest(int(self.left_counts[idx]), int(self.totals[idx]), 0.5).pvalue

    def relevant_row(self, idx: int) -> Tuple[int, int, float]:
        return int(self.left_counts[idx]), int(self.totals[idx]), float(self.pvalues[idx])


class MatchingPennies2:
    """
    Python port of the MATLAB classdef MatchingPennies2.

    This algorithm:
      - Maintains n-gram tables for n=1..N over (choice,reward) tokens (4 symbols).
      - For each new observation, updates each n-gram context row.
      - On sampling, finds the relevant row (current context) for each n,
        chooses the row with smallest p-value, and if p <= alpha uses it to pick a choice.

    NOTE: The MATLAB code returns:
        trial_types((left/total > 0.5) + 1)
      where trial_types = ["L","R"].
      So if left/total > 0.5 => chooses "R", else chooses "L".
      This is preserved here for fidelity. If you want the *predicted opponent choice*
      instead, flip `invert_prediction=False` below.
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
        mp1: Optional[object] = None,  # optional: if you have a separate MatchingPennies1
    ):
        self.trials_back = int(N)
        self.alpha = float(alpha)
        self.invert_prediction = bool(invert_prediction)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Histories
        self.choice_history: List[Choice] = []
        self.reward_history: List[int] = []
        self.choice_reward_combination: List[str] = []
        self.tt_history: List[Choice] = []

        # ngram tables for n=1..N (index 0 unused for convenience)
        self.ngram: List[Optional[NGramTable]] = [None]
        for n in range(1, self.trials_back + 1):
            self.ngram.append(NGramTable.initialize(n))

        # Optional: carry the original composition (obj.mp1 = tt.MatchingPennies1(...))
        self.mp1 = mp1

    def update(self, last_choice: Choice, last_reward: int) -> None:
        if last_choice not in ("L", "R", "M"):
            raise ValueError(f"last_choice must be 'L', 'R', or 'M', got {last_choice!r}")
        last_reward_int = int(last_reward)
        if last_reward_int not in (0, 1):
            raise ValueError(f"last_reward must be 0/1, got {last_reward!r}")

        if self.mp1 is not None and hasattr(self.mp1, "update"):
            self.mp1.update(last_choice, last_reward_int)

        last_trial = f"{last_choice} {last_reward_int}"

        # MATLAB: if last_trial ~= "M 0", then update ngrams
        if last_trial != self.MISS_TOKEN:
            for trial_back in range(1, self.trials_back + 1):
                self._update_ngram(trial_back=trial_back, last_trial=last_trial)

        self.choice_history.append(last_choice)
        self.reward_history.append(last_reward_int)
        self.choice_reward_combination.append(last_trial)

    def _update_ngram(self, trial_back: int, last_trial: str) -> None:
        # MATLAB condition: if trial_back < length(choice_history)
        # meaning: only update once we have at least trial_back prior tokens (excluding current)
        if trial_back < len(self.choice_history):
            idx = self._history_to_index(trial_back)
            table = self.ngram[trial_back]
            assert table is not None
            table.update(idx=idx, last_choice=last_trial[0])  # first char 'L'/'R'

    def _history_to_index(self, trial_back: int) -> int:
        # MATLAB:
        #   trial_combination = choice_reward_combination;
        #   trial_combination(trial_combination=="M 0") = [];
        #   trial_combination = trial_combination(end-(trial_back-1):end);
        #   barcode: L0->0, R0->1, L1->2, R1->3
        #   idx = dot(4.^(0:numel-1), barcode) + 1
        #
        # Python: use 0-based idx (so omit +1).
        trial_combination = [t for t in self.choice_reward_combination if t != self.MISS_TOKEN]
        if len(trial_combination) < trial_back:
            raise RuntimeError("Not enough non-miss history to compute index.")

        recent = trial_combination[-trial_back:]  # oldest..newest among last trial_back
        # MATLAB uses 4^(0..n-1) with barcode ordered as given.
        # Their code sets trial_combination = end-(n-1):end, then dot(4^(0..), barcode)
        # => barcode[0] is weighted by 4^0, i.e. the OLDEST of that slice gets 4^0.
        # We'll match that exactly: idx = sum(barcode[i] * 4^i).
        barcode = [self.TOKEN_TO_DIGIT[t] for t in recent]
        idx0 = int(sum(d * (4 ** i) for i, d in enumerate(barcode)))
        return idx0

    def relevant_rows(self) -> List[Tuple[int, int, float]]:
        rows: List[Tuple[int, int, float]] = []
        for n in range(1, self.trials_back + 1):
            if n < len(self.choice_history):
                idx = self._history_to_index(n)
                table = self.ngram[n]
                assert table is not None
                rows.append(table.relevant_row(idx))
        if self.mp1 is not None and hasattr(self.mp1, "relevant_rows"):
            # Expecting a list of (left, total, pvalue)-like tuples
            rows.extend(self.mp1.relevant_rows())
        return rows

    def sample(self) -> Choice:
        # MATLAB:
        # if trials_back < length(choice_history) then consult rows else random
        if self.trials_back < len(self.choice_history):
            rows = self.relevant_rows()
            if not rows:
                choice = self._random_choice()
            else:
                pvals = np.array([r[2] for r in rows], dtype=float)
                # handle NaNs (unseen contexts): treat as +inf so they won't be chosen
                pvals = np.where(np.isnan(pvals), np.inf, pvals)
                best_idx = int(np.argmin(pvals))
                left, total, pval = rows[best_idx]

                if not np.isfinite(pval) or pval > self.alpha or total <= 0:
                    choice = self._random_choice()
                else:
                    p_left = left / total
                    # MATLAB: trial_types((p_left > 0.5) + 1) with ["L","R"]
                    # => if p_left>0.5 choose "R" else choose "L"
                    predicted = "R" if (p_left > 0.5) else "L"
                    if self.invert_prediction:
                        # invert (useful if you're trying to *beat* predicted opponent choice)
                        choice = "L" if predicted == "R" else "R"
                    else:
                        choice = predicted
        else:
            choice = self._random_choice()

        self.tt_history.append(choice)
        return choice

    def _random_choice(self) -> Choice:
        return self.TRIAL_TYPES[int(self.rng.integers(0, 2))]
    
    
    def to_dict(self):
        """
        Return a lightweight, JSON-serializable summary of the algo state.
        Avoid dumping full n-gram tables (they grow as 4^N).
        """

        summary = {
            "n_trials_seen": len(self.choice_history),
            "trials_back": self.trials_back,
            "alpha": self.alpha,
        }

        # Try to compute the currently most informative n-back
        if self.trials_back < len(self.choice_history):
            rows = self.relevant_rows()

            if rows:
                # rows = [(left, total, pval), ...]
                pvals = []
                for r in rows:
                    pvals.append(r[2] if r[2] is not None else float("inf"))

                best_idx = int(np.argmin(pvals))
                best_row = rows[best_idx]

                left, total, pval = best_row
                p_left = left / total if total > 0 else None

                summary.update({
                    "best_n_back": best_idx + 1,
                    "best_pvalue": pval,
                    "p_left_estimate": p_left,
                })

        return summary


# --- tiny usage example ---
if __name__ == "__main__":
    algo = MatchingPennies2(N=3, alpha=0.05, invert_prediction=True)

    # feed opponent observations (choice, reward)
    observations = [("L", 1), ("R", 0), ("L", 0), ("L", 1), ("R", 1)]
    for c, r in observations:
        algo.update(c, r)
        print("algo choice:", algo.sample())