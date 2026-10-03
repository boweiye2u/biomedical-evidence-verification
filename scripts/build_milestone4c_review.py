"""Build the structured review after the Milestone 4C sample IDs are frozen."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

ART = Path(os.environ.get("RAG_ROOT", str(Path.home() / "rag")))
RUN = ART / "runs/milestone4c-analysis-v1"

# These judgments were made only after analysis-sample-ids.json was written.
# Categories describe visible ranking behavior; they do not diagnose training causality.
REVIEWS = {
    "82": ("degradation", "mechanism_or_relation_specificity", "Mined rankings retain cardiac/BMP/reprogramming themes but demote the exact multi-factor cardiovascular-progenitor study."),
    "244": ("degradation", "broad_topic_substitution", "Mined rankings favor broad macrophage/inflammation and KLF2 papers while losing the TLR monocyte-to-macrophage/dendritic-cell paper."),
    "989": ("degradation", "mechanism_or_relation_specificity", "Mined rankings retain neural progenitor and stem-cell papers but demote the study tied to isolation-free symmetrical self-renewal."),
    "304": ("degradation", "lexical_entity_collision", "Mined results include Dishevelled 3 and generic transcription/chromatin papers, consistent with ambiguity around the short symbol DMS3."),
    "1261": ("degradation", "polarity_or_relation_insensitivity", "The claim reverses generation/removal wording; mined results shift toward generic oncogene and genome-stability papers and demote the BCR/ABL ROS repair study."),
    "227": ("degradation", "broad_topic_substitution", "Mined results emphasize general residual disease, relapse, and resistance rather than the EGFR-targeted-therapy resistance review."),
    "1312": ("degradation", "lexical_entity_collision", "Mined results favor unrelated transcription factors and B-cell biology, while the TFEB antimicrobial host-defense study falls below them."),
    "1220": ("degradation", "mechanism_or_relation_specificity", "Mined results retain cancer and genome-stability themes but demote the precise BCR/ABL–ROS–double-strand-break study."),
    "428": ("degradation", "lexical_entity_collision", "Mined results broaden from FOXO to other forkhead-family genes and functions, losing the stem-cell-homeostasis review from the top ten."),
    "280": ("degradation", "mechanism_or_relation_specificity", "The exact comparative megakaryocyte/erythroblast expression study is displaced by broader transcription and platelet biology."),
    "28": ("degradation", "polarity_or_relation_insensitivity", "The claim says the Th2 environment impedes SLE, whereas the known paper says it can promote lupus nephritis; mined results retain SLE topics but demote that relation-bearing paper."),
    "1002": ("degradation", "mechanism_or_relation_specificity", "Mined results retain promoter/transcription themes but lose the nuclear-receptor histone-methylation paper associated with RA-induced active-promoter hallmarks."),
    "66": ("degradation", "benchmark_positive_semantic_gap", "The known positive title concerns antiretroviral hepatotoxicity rather than the stated AZT–ribavirin anemia relation, so ranking movement is difficult to interpret mechanistically."),
    "550": ("degradation", "mechanism_or_relation_specificity", "Mined results retain T-cell signaling but substitute costimulation, trafficking, and exhaustion papers for the CD3 clustering/conformational-change study."),
    "426": ("degradation", "lexical_entity_collision", "Mined rankings broaden from FOXO apoptosis/homeostasis to other forkhead-family genes and functions."),
    "1407": ("degradation", "lexical_entity_collision", "The β1/Ketel query attracts β1-integrin and generic microtubule papers; both known positives are strongly demoted, one beyond rank 100."),
    "361": ("degradation", "mechanism_or_relation_specificity", "Mined results retrieve closely related B-cell migration and oxysterol papers but demote the review that joins these concepts in the early antibody response."),
    "157": ("improvement", "no_clear_failure", "Both trained arms keep BLM/RecQ papers near the top; mined is slightly better than random but below zero-shot rank 1."),
    "176": ("degradation", "benchmark_positive_semantic_gap", "The known positive title is a Wolfram-syndrome insulin-secretion paper rather than a direct BiP marker paper; relevant-looking ER-stress papers appear above it."),
    "11": ("near_tie", "benchmark_positive_semantic_gap", "The known positive is indirect for a 4-PBA/ER-stress claim; both trained arms rank other ER-stress papers above it."),
    "17": ("degradation", "lexical_entity_collision", "Mined results include 53BP1 papers, suggesting the numeric token 53 was treated as an entity; the known positive is also indirect for perinatal mortality."),
    "761": ("degradation", "benchmark_positive_semantic_gap", "Mined rankings retrieve several directly MeCP2-related maturation papers, while the designated structural-homeostasis positive is only indirectly linked and falls to ranks 79–88."),
    "62": ("near_tie", "benchmark_positive_semantic_gap", "The designated Wolfram-syndrome paper is indirect for ATF4 as a general ER-stress marker, and mined retrieves other stress/transcription papers while losing it beyond rank 100."),
    "703": ("degradation", "no_clear_failure", "The known plant-polarity paper remains rank 1–2 across systems; the small NDCG change does not reveal a clear failure."),
    "378": ("near_tie", "mixed_positive_reordering", "With five known positives, mined promotes some hypothalamic/glutamate papers and demotes others; the aggregate NDCG difference is near zero."),
    "898": ("degradation", "mixed_positive_reordering", "The direct TREX1/STING paper stays rank 1, while the second positive is demoted; the failure is concentrated in one of two positives."),
    "1040": ("near_tie", "mixed_positive_reordering", "One H2A.Z positive remains at rank 7 and the other is demoted; the net mined-versus-random NDCG change is negligible."),
    "1253": ("improvement", "exact_positive_promotion", "Mined moves the non-coding-RNA hematopoiesis paper from zero-shot rank 65 to rank 1 in all seeds."),
    "655": ("improvement", "polarity_or_relation_insensitivity", "Mined moves the FNDC5/irisin paper to rank 1, but the paired sample contains opposite increase/reduce claim wording tied to the same positive, limiting polarity interpretation."),
    "654": ("improvement", "polarity_or_relation_insensitivity", "Mined moves the FNDC5/irisin paper to rank 1, but the paired sample contains opposite increase/reduce claim wording tied to the same positive, limiting polarity interpretation."),
    "211": ("improvement", "exact_positive_promotion", "Mined restores the designated COPI paper to rank 1 across seeds after random training demoted it."),
    "673": ("improvement", "exact_positive_promotion", "Mined moves the LDL genome-wide association paper to rank 1 across seeds."),
    "679": ("near_tie", "exact_positive_promotion", "Both trained arms move the nuclear-receptor histone-methylation paper to rank 1; mined and random are tied."),
    "572": ("improvement", "benchmark_positive_semantic_gap", "Mined improves the PTPσ spinal-cord-injury paper relative to random, but that positive does not directly express the pDC/chronic-infection claim."),
    "1067": ("degradation", "mechanism_or_relation_specificity", "Both arms improve greatly over zero-shot; mined trails random because the exact Pif1 G-quadruplex paper is rank 3 rather than rank 2."),
    "782": ("near_tie", "no_clear_failure", "The autoimmune-myocarditis positive stays rank 1–2 across systems; no stable mined-versus-random failure is visible."),
    "951": ("improvement", "exact_positive_promotion", "Mined consistently moves the Piezo1 mobility paper to rank 1."),
    "541": ("improvement", "polarity_or_relation_insensitivity", "The claim says unrelated, yet all systems retrieve strong hypothalamic glutamate/energy-balance papers; mined improves one positive while demoting others."),
    "1397": ("improvement", "benchmark_positive_semantic_gap", "Mined promotes the p16 reporter-model paper, although the title does not directly establish the specific oral-lesion wound-response relation."),
    "1398": ("improvement", "benchmark_positive_semantic_gap", "Mined promotes the p16 reporter-model paper, although the title does not directly establish the stated degradation/CDKN2A relation."),
}


def main():
    sample = json.loads((RUN / "analysis-sample-ids.json").read_text())
    candidates = json.loads((RUN / "review-candidates.json").read_text())
    sample_ids = sample["query_ids"]
    assert list(REVIEWS) == sample_ids, "review order/coverage must exactly match frozen sample"
    by_id = {item["query_id"]: item for item in candidates}
    assert set(by_id) == set(sample_ids) and len(sample_ids) == 40

    records = []
    for query_id in sample_ids:
        outcome, category, note = REVIEWS[query_id]
        candidate = by_id[query_id]
        records.append({
            "query_id": query_id,
            "selection_reason": candidate["selection_reason"],
            "claim": candidate["claim"],
            "outcome": outcome,
            "primary_category": category,
            "analyst_note": note,
            "mined_minus_random_ndcg_at_10": candidate["differences"]["NDCG@10"]["mined_minus_random"],
            "positive_ranks": candidate["positive_ranks"],
        })

    result = {
        "scope": "structured analyst-assisted review of IDs frozen before category assignment",
        "reviewer": "Codex analyst-assisted review; not independent blinded human annotation",
        "causal_limit": "Categories describe observed retrieval outputs and cannot identify why training changed them.",
        "category_definitions": {
            "mechanism_or_relation_specificity": "Correct broad topic, but the exact relation or mechanism is displaced.",
            "broad_topic_substitution": "Broadly related disease/process papers displace the designated positive.",
            "lexical_entity_collision": "A shared token, short symbol, family name, or number attracts a different entity.",
            "polarity_or_relation_insensitivity": "Rankings do not reliably distinguish opposite or altered claim relations.",
            "benchmark_positive_semantic_gap": "The designated BEIR positive is only indirectly connected to the claim from visible title/text evidence.",
            "mixed_positive_reordering": "Multiple positives move in different directions.",
            "exact_positive_promotion": "The designated positive is consistently promoted; this is an improvement case.",
            "no_clear_failure": "The ranking change is small or does not support a specific category.",
        },
        "outcome_counts": dict(Counter(item["outcome"] for item in records)),
        "category_counts": dict(Counter(item["primary_category"] for item in records)),
        "records": records,
    }
    output = RUN / "structured-analyst-assisted-review.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "outcome_counts": result["outcome_counts"], "category_counts": result["category_counts"]}, indent=2))


if __name__ == "__main__":
    main()
