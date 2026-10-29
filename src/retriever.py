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
            "iphone" in q
            or "ios" in q
            or "apple" in q
        ):
            return "ios_pairing"

        if (
            "android" in q
            or "samsung" in q
        ):
            return "android_pairing"

        if (
            "pair" in q
            or "pairing" in q
            or "connect" in q
            or "connecting" in q
        ):
            return "general"

        if (
            "noise canceling" in q
            or "noise cancelling" in q
            or "noise cancellation" in q
            or "ambient sound" in q
            or "nc/amb" in q
        ):
            return "noise_canceling"

        if (
            "how many devices" in q
            or "number of devices" in q
            or "maximum devices" in q
            or ("how many" in q and "device" in q)
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

        elif intent == "ios_pairing":
            if "iphone" in section or "ios" in section:
                score += 5.0
            if "apple devices" in section:
                score += 4.0
            if "pairing and connecting" in section:
                score += 2.0
            if "le audio" in section:
                score -= 8.0

        elif intent == "le_audio":
            if "le audio" in section:
                score += 6.0

        elif intent == "noise_canceling":
            if (
                "using the noise canceling function" in section
                or "using noise canceling function" in section
            ):
                score += 6.0
            if "ambient sound" in section:
                score -= 2.0

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
        candidate_count = max(top_k * 4, 12)

        candidates = self.vector_store.search(
            question=question,
            top_k=candidate_count,
            chapter=chapter,
        )

        if not candidates:
            return []

        intent = self._classify_question(question)

        for item in candidates:
            distance = float(item.get("distance", 1.0))
            item["_rerank_score"] = (
                -distance + self._section_score(item, intent)
            )

        candidates.sort(
            key=lambda item: item["_rerank_score"],
            reverse=True,
        )

        if intent == "android_pairing":
            ordinary = [
                item for item in candidates
                if "le audio" not in self._get_metadata_text(item)
            ]
            if ordinary:
                candidates = ordinary

        # Remove exact duplicate chunks and repeated section/page entries.
        results = []
        seen_text = set()
        seen_location = set()

        for item in candidates:
            text = " ".join(item.get("text", "").lower().split())
            metadata = item.get("metadata", {})

            location = (
                str(metadata.get("section", "")).strip().lower(),
                metadata.get("page_start"),
                metadata.get("page_end"),
            )

            if text in seen_text or location in seen_location:
                continue

            seen_text.add(text)
            seen_location.add(location)

            cleaned = dict(item)
            cleaned.pop("_rerank_score", None)
            results.append(cleaned)

            if len(results) >= top_k:
                break

        return results
