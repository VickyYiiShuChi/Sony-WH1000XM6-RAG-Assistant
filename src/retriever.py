
from src.config import TOP_K


class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    @staticmethod
    def _get_metadata_text(item):
        metadata = item.get("metadata", {})

        return (
            f"{metadata.get('chapter', '')} "
            f"{metadata.get('section', '')}"
        ).lower()

    @staticmethod
    def _classify_question(question):
        q = question.lower()

        # Check specific operations before general keywords.
        if (
            "factory reset" in q
            or "factory settings" in q
            or "initialize" in q
            or "initialise" in q
            or "restore factory" in q
        ):
            return "initialize"

        if "reset" in q or "resetting" in q:
            return "reset"

        if "le audio" in q:
            return "le_audio"

        if (
            "pair" in q
            or "pairing" in q
            or "connect" in q
            or "connecting" in q
        ):
            if "android" in q:
                return "android_pairing"

        if (
            "noise canceling" in q
            or "noise cancelling" in q
            or "ambient sound" in q
            or "nc/amb" in q
        ):
            return "noise_canceling"

        if (
            "how many devices" in q
            or "number of devices" in q
            or "maximum devices" in q
            or "how many" in q and "device" in q
        ):
            return "paired_device_limit"

        return "general"

    @staticmethod
    def _section_score(item, intent):
        metadata = item.get("metadata", {})
        section = str(metadata.get("section", "")).lower()
        text = item.get("text", "").lower()

        score = 0.0

        if intent == "reset":
            if "resetting the headset" in section:
                score += 5.0
            if "initializing the headset" in section:
                score -= 8.0
            if "restore factory settings" in section:
                score -= 8.0

        elif intent == "initialize":
            if "initializing the headset" in section:
                score += 5.0
            if "restore factory settings" in section:
                score += 4.0
            if "resetting the headset" in section:
                score -= 8.0

        elif intent == "android_pairing":
            if "android smartphone" in section:
                score += 4.0
            if "le audio" in section:
                score -= 8.0

            if "pairing and connecting" in section:
                score += 2.0

        elif intent == "le_audio":
            if "le audio" in section:
                score += 6.0

        elif intent == "noise_canceling":
            if "using noise canceling function" in section:
                score += 4.0
            if "ambient sound" in section:
                score += 3.0

        elif intent == "paired_device_limit":
            if "pairing" in section or "connecting" in section:
                score += 1.0
            if (
                "up to 8 devices" in text
                or "up to eight devices" in text
                or "maximum of 8 devices" in text
            ):
                score += 5.0

        return score

    def retrieve(self, question, top_k=TOP_K, chapter=None):
        # Retrieve extra candidates before reranking.
        candidate_count = max(top_k * 4, 12)

        candidates = self.vector_store.search(
            question=question,
            top_k=candidate_count,
            chapter=chapter,
        )

        if not candidates:
            return []

        intent = self._classify_question(question)

        # Preserve semantic relevance while using section information
        # to distinguish similar procedures.
        for item in candidates:
            distance = item.get("distance", 1.0)

            item["_rerank_score"] = (
                -float(distance)
                + self._section_score(item, intent)
            )

        candidates.sort(
            key=lambda item: item["_rerank_score"],
            reverse=True,
        )

        # For ordinary Android pairing, do not return LE Audio
        # sections when enough ordinary pairing results exist.
        if intent == "android_pairing":
            ordinary_candidates = [
                item
                for item in candidates
                if "le audio" not in self._get_metadata_text(item)
            ]

            if ordinary_candidates:
                candidates = ordinary_candidates

        # Remove internal reranking scores from returned results.
        results = candidates[:top_k]

        for item in results:
            item.pop("_rerank_score", None)

        return results
