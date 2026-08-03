"""Temperature-scaled weighted sampling for word and character models."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Sequence

from .cleaner import (
    clean_text,
    detokenize,
    is_lexical_token,
    normalize_character_prompt,
    tokenize,
)
from .database import BOS_TOKEN, EOS_TOKEN, MarkovDatabase, ModelInfo
from .exceptions import ConfigurationError, GenerationError


_TOPIC_STOPWORDS = frozenset(
    {
        "acaba", "ama", "ancak", "artık", "ben", "bir", "biz", "bu",
        "da", "daha", "de", "diye", "en", "fakat", "gibi", "hem", "için",
        "ile", "ise", "ki", "mi", "mı", "mu", "mü", "ne", "neden", "o",
        "sen", "siz", "şu", "ve", "veya", "ya", "çok",
    }
)
_NOISY_OPENERS = frozenset({"(", "[", "♪", "♫"})


def weighted_sample(
    candidates: Sequence[tuple[str, int]],
    *,
    temperature: float,
    random_source: random.Random,
) -> str:
    """Sample counts after stable ``count ** (1 / temperature)`` scaling."""

    if not candidates:
        raise GenerationError("cannot sample from an empty transition set")
    if not math.isfinite(temperature) or temperature < 0:
        raise ConfigurationError("temperature must be a finite number >= 0")
    if temperature == 0:
        # Lexical tie-breaking keeps seeded and unseeded greedy output stable.
        return max(candidates, key=lambda item: (item[1], item[0]))[0]

    log_counts = [math.log(frequency) for _, frequency in candidates]
    maximum = max(log_counts)
    # Subtract before dividing so a tiny positive temperature cannot overflow
    # the largest log-count to +inf. All exponents are then <= 0.
    weights = [
        math.exp((log_count - maximum) / temperature)
        for log_count in log_counts
    ]
    threshold = random_source.random() * sum(weights)
    cumulative = 0.0
    for (token, _), weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if cumulative > threshold:
            return token
    return candidates[-1][0]


class MarkovGenerator:
    """Generate normalized Turkish text or pseudo-words from a database."""

    def __init__(self, database: MarkovDatabase, *, seed: int | None = None) -> None:
        self.database = database
        self.random = random.Random(seed)
        self._candidate_cache: dict[
            tuple[int, tuple[str, ...]], list[tuple[str, int]]
        ] = {}

    def _candidates(
        self, model: ModelInfo, context: Sequence[str]
    ) -> list[tuple[str, int]]:
        cache_key = (model.model_id, tuple(context))
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        candidates = self.database.exact_transitions(model.model_id, context)
        if candidates:
            self._candidate_cache[cache_key] = candidates
            return candidates
        last_token = context[-1]
        if last_token != BOS_TOKEN:
            candidates = self.database.backoff_transitions(model.model_id, last_token)
        self._candidate_cache[cache_key] = candidates
        return candidates

    def _topic_starts(
        self, model: ModelInfo, prompt_tokens: Sequence[str]
    ) -> list[tuple[str, int]]:
        """Find prompt words that genuinely occur at a learned sentence start."""

        unique: list[str] = []
        for token in prompt_tokens:
            if (
                is_lexical_token(token)
                and token not in _TOPIC_STOPWORDS
                and token not in unique
            ):
                unique.append(token)
        if not unique:
            unique = [
                token
                for token in prompt_tokens
                if is_lexical_token(token) and token not in unique
            ]
        bos_context = (BOS_TOKEN,) * model.order
        starts: list[tuple[str, int]] = []
        for token in unique:
            frequency = self.database.transition_frequency(
                model.model_id, bos_context, token
            )
            if frequency:
                # A rarer prompt word normally carries more topic information
                # than a corpus-wide word such as "gün". Inverse square-root
                # weighting preserves variety without letting common words win.
                topic_weight = max(1, round(1_000_000 / math.sqrt(frequency)))
                starts.append((token, topic_weight))
        # Put the most distinctive (rarest) topic words first. Restarts cycle
        # through this list, preventing every sentence from repeating one word.
        return sorted(starts, key=lambda item: (-item[1], item[0]))

    @staticmethod
    def _context(order: int, prefix: Sequence[str]) -> deque[str]:
        padding = max(0, order - len(prefix))
        return deque(
            [BOS_TOKEN] * padding + list(prefix[-order:]),
            maxlen=order,
        )

    def generate_word_text(
        self,
        prompt: str,
        *,
        length: int,
        temperature: float = 1.0,
    ) -> str:
        """Generate ``length`` new lexical tokens after a normalized prompt."""

        if length < 1:
            raise ConfigurationError("length must be at least 1")
        if not math.isfinite(temperature) or temperature < 0:
            raise ConfigurationError("temperature must be a finite number >= 0")
        model = self.database.model("word")
        cleaned_prompt = clean_text(prompt)
        prompt_tokens = tokenize(cleaned_prompt, already_clean=True)
        context = self._context(model.order, prompt_tokens)
        output = list(prompt_tokens)
        topic_starts = self._topic_starts(model, prompt_tokens)
        topic_restarts: list[tuple[str, ...]] = []
        if topic_starts:
            anchor = topic_starts[0][0]
            topic_restarts.append((anchor,))
            if model.order > 1:
                for index in range(len(prompt_tokens) - model.order + 1):
                    phrase = tuple(prompt_tokens[index : index + model.order])
                    if (
                        anchor in phrase
                        and phrase not in topic_restarts
                        and self._candidates(model, phrase)
                    ):
                        topic_restarts.append(phrase)
        topic_restart_index = 0
        seen_ngrams = {
            tuple(output[index : index + 3])
            for index in range(max(0, len(output) - 2))
        }
        generated_words = 0
        attempts = 0
        max_attempts = max(100, length * 25)

        while generated_words < length and attempts < max_attempts:
            attempts += 1
            candidates = self._candidates(model, tuple(context))
            cleaner_candidates = [
                item for item in candidates if item[0] not in _NOISY_OPENERS
            ]
            if cleaner_candidates:
                candidates = cleaner_candidates
            if len(output) >= 2:
                nonrepeating = [
                    item
                    for item in candidates
                    if tuple(output[-2:] + [item[0]]) not in seen_ngrams
                ]
                if nonrepeating:
                    candidates = nonrepeating
            if not candidates:
                if not output:
                    raise GenerationError("word model has no valid sentence start")
                shown_prompt = detokenize(prompt_tokens) or prompt
                raise GenerationError(
                    f"no transition found for prompt/context {shown_prompt!r}"
                )
            next_token = weighted_sample(
                candidates,
                temperature=temperature,
                random_source=self.random,
            )
            if next_token == EOS_TOKEN:
                if not topic_restarts:
                    # Starting from the entire corpus here can mean reading
                    # millions of rows and abruptly switching to an unrelated
                    # book/subtitle. A shorter focused result is preferable.
                    break
                remaining = length - generated_words
                eligible_restarts = [
                    phrase for phrase in topic_restarts if len(phrase) <= remaining
                ]
                phrase = eligible_restarts[
                    topic_restart_index % len(eligible_restarts)
                ]
                topic_restart_index += 1
                context = self._context(model.order, [])
                for restart_token in phrase:
                    output.append(restart_token)
                    context.append(restart_token)
                    generated_words += int(is_lexical_token(restart_token))
                    if len(output) >= 3:
                        seen_ngrams.add(tuple(output[-3:]))
                continue
            if next_token == BOS_TOKEN:
                context = self._context(model.order, [])
                continue
            output.append(next_token)
            context.append(next_token)
            if len(output) >= 3:
                seen_ngrams.add(tuple(output[-3:]))
            if is_lexical_token(next_token):
                generated_words += 1

        if generated_words < length and attempts >= max_attempts:
            raise GenerationError(
                "generation stopped before reaching the requested word count; "
                "the model contains too many non-word cycles"
            )
        return detokenize(output)

    def generate_word(
        self,
        prompt: str,
        *,
        length: int,
        temperature: float = 1.0,
    ) -> str:
        """Generate a pseudo-Turkish word, treating ``prompt`` as its prefix."""

        if length < 1:
            raise ConfigurationError("length must be at least 1")
        if not math.isfinite(temperature) or temperature < 0:
            raise ConfigurationError("temperature must be a finite number >= 0")
        model = self.database.model("char")
        try:
            prefix = normalize_character_prompt(prompt)
        except ValueError as exc:
            raise GenerationError(str(exc)) from exc
        output = list(prefix)
        context = self._context(model.order, output)

        for _ in range(length):
            candidates = self._candidates(model, tuple(context))
            if not candidates:
                shown_prefix = prefix or "<empty>"
                raise GenerationError(
                    f"no character transition found for prefix {shown_prefix!r}"
                )
            next_token = weighted_sample(
                candidates,
                temperature=temperature,
                random_source=self.random,
            )
            if next_token == EOS_TOKEN:
                break
            if next_token == BOS_TOKEN:
                continue
            output.append(next_token)
            context.append(next_token)
        return "".join(output)

    def generate(
        self,
        mode: str,
        prompt: str,
        *,
        length: int,
        temperature: float = 1.0,
    ) -> str:
        """Dispatch to a word-level or character-level generator."""

        if mode == "word":
            return self.generate_word_text(
                prompt, length=length, temperature=temperature
            )
        if mode == "char":
            return self.generate_word(prompt, length=length, temperature=temperature)
        raise ConfigurationError("generation mode must be word or char")
