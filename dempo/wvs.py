from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml


COUNTRY_INFO: Dict[str, Dict[str, str]] = {
    "AND": {"country": "Andorra", "region": "Europe"},
    "ARG": {"country": "Argentina", "region": "Americas"},
    "ARM": {"country": "Armenia", "region": "Asia"},
    "AUS": {"country": "Australia", "region": "Oceania"},
    "BGD": {"country": "Bangladesh", "region": "Asia"},
    "BOL": {"country": "Bolivia", "region": "Americas"},
    "BRA": {"country": "Brazil", "region": "Americas"},
    "CAN": {"country": "Canada", "region": "Americas"},
    "CHL": {"country": "Chile", "region": "Americas"},
    "CHN": {"country": "China", "region": "Asia"},
    "COL": {"country": "Colombia", "region": "Americas"},
    "CYP": {"country": "Cyprus", "region": "Europe"},
    "CZE": {"country": "Czech Republic", "region": "Europe"},
    "DEU": {"country": "Germany", "region": "Europe"},
    "ECU": {"country": "Ecuador", "region": "Americas"},
    "EGY": {"country": "Egypt", "region": "Africa"},
    "ETH": {"country": "Ethiopia", "region": "Africa"},
    "GBR": {"country": "Great Britain", "region": "Europe"},
    "GRC": {"country": "Greece", "region": "Europe"},
    "GTM": {"country": "Guatemala", "region": "Americas"},
    "HKG": {"country": "Hong Kong", "region": "Asia"},
    "IDN": {"country": "Indonesia", "region": "Asia"},
    "IND": {"country": "India", "region": "Asia"},
    "IRN": {"country": "Iran", "region": "Asia"},
    "IRQ": {"country": "Iraq", "region": "Asia"},
    "JOR": {"country": "Jordan", "region": "Asia"},
    "JPN": {"country": "Japan", "region": "Asia"},
    "KAZ": {"country": "Kazakhstan", "region": "Asia"},
    "KEN": {"country": "Kenya", "region": "Africa"},
    "KGZ": {"country": "Kyrgyzstan", "region": "Asia"},
    "KOR": {"country": "South Korea", "region": "Asia"},
    "LBN": {"country": "Lebanon", "region": "Asia"},
    "LBY": {"country": "Libya", "region": "Africa"},
    "MAC": {"country": "Macao", "region": "Asia"},
    "MAR": {"country": "Morocco", "region": "Africa"},
    "MDV": {"country": "Maldives", "region": "Asia"},
    "MEX": {"country": "Mexico", "region": "Americas"},
    "MMR": {"country": "Myanmar", "region": "Asia"},
    "MNG": {"country": "Mongolia", "region": "Asia"},
    "MYS": {"country": "Malaysia", "region": "Asia"},
    "NGA": {"country": "Nigeria", "region": "Africa"},
    "NIC": {"country": "Nicaragua", "region": "Americas"},
    "NIR": {"country": "Northern Ireland", "region": "Europe"},
    "NLD": {"country": "Netherlands", "region": "Europe"},
    "NZL": {"country": "New Zealand", "region": "Oceania"},
    "PAK": {"country": "Pakistan", "region": "Asia"},
    "PER": {"country": "Peru", "region": "Americas"},
    "PHL": {"country": "Philippines", "region": "Asia"},
    "PRI": {"country": "Puerto Rico", "region": "Americas"},
    "ROU": {"country": "Romania", "region": "Europe"},
    "RUS": {"country": "Russia", "region": "Europe"},
    "SGP": {"country": "Singapore", "region": "Asia"},
    "SRB": {"country": "Serbia", "region": "Europe"},
    "SVK": {"country": "Slovakia", "region": "Europe"},
    "THA": {"country": "Thailand", "region": "Asia"},
    "TJK": {"country": "Tajikistan", "region": "Asia"},
    "TUN": {"country": "Tunisia", "region": "Africa"},
    "TUR": {"country": "Turkey", "region": "Asia"},
    "TWN": {"country": "Taiwan", "region": "Asia"},
    "UKR": {"country": "Ukraine", "region": "Europe"},
    "URY": {"country": "Uruguay", "region": "Americas"},
    "USA": {"country": "United States", "region": "Americas"},
    "UZB": {"country": "Uzbekistan", "region": "Asia"},
    "VEN": {"country": "Venezuela", "region": "Americas"},
    "VNM": {"country": "Vietnam", "region": "Asia"},
    "ZWE": {"country": "Zimbabwe", "region": "Africa"},
}


def _option_map(options: List[str]) -> Dict[str, str]:
    labels = ["A", "B", "C", "D"]
    return {label: option for label, option in zip(labels, options)}


def _question(code: int, section: str, prompt: str, options: List[str]) -> dict:
    if len(options) != 4:
        raise ValueError(f"WVS filtered question Q{code} must have exactly 4 options, got {len(options)}")
    return {
        "question_id": code,
        "question_code": f"Q{code}",
        "section": section,
        "question_text": prompt,
        "options": _option_map(options),
        "value_codes": {"A": 1, "B": 2, "C": 3, "D": 4},
        "subjective": True,
        "country_specific": False,
    }


def build_subjective_questions() -> List[dict]:
    questions: List[dict] = []

    importance_options = [
        "Very important",
        "Rather important",
        "Not very important",
        "Not at all important",
    ]
    # Keep only public / political value questions, not personal-life salience items.
    questions.append(
        _question(
            4,
            "social_values",
            "How important is politics in your life?",
            importance_options,
        )
    )

    agree4 = ["Strongly agree", "Agree", "Disagree", "Strongly disagree"]
    for qid, statement in [
        (28, "When a mother works for pay, the children suffer."),
        (29, "On the whole, men make better political leaders than women do."),
        (30, "A university education is more important for a boy than for a girl."),
        (31, "On the whole, men make better business executives than women do."),
        (32, "Being a housewife is just as fulfilling as working for pay."),
        (169, "Whenever science and religion conflict, religion is always right."),
        (170, "The only acceptable religion is my religion."),
    ]:
        questions.append(
            _question(
                qid,
                "values_and_norms",
                f"Please indicate how strongly you agree or disagree with the following statement: \"{statement}\"",
                agree4,
            )
        )

    trust_options = [
        "Trust completely",
        "Trust somewhat",
        "Do not trust very much",
        "Do not trust at all",
    ]
    for qid, group in [
        (61, "people you meet for the first time"),
        (62, "people of another religion"),
        (63, "people of another nationality"),
    ]:
        questions.append(
            _question(
                qid,
                "social_trust",
                f"How much do you trust {group}?",
                trust_options,
            )
        )

    confidence_options = [
        "A great deal of confidence",
        "Quite a lot of confidence",
        "Not very much confidence",
        "None at all",
    ]
    confidence_items = [
        (64, "religious institutions"),
        (65, "the armed forces"),
        (66, "the press"),
        (67, "television"),
        (68, "labor unions"),
        (69, "the police"),
        (70, "the courts"),
        (71, "the government"),
        (72, "political parties"),
        (73, "parliament"),
        (74, "the civil service"),
        (75, "universities"),
        (76, "elections"),
        (77, "major companies"),
        (78, "banks"),
        (79, "environmental organizations"),
        (80, "women's organizations"),
        (81, "charitable or humanitarian organizations"),
        (83, "the United Nations"),
        (84, "the International Monetary Fund (IMF)"),
        (85, "the International Criminal Court (ICC)"),
        (86, "the North Atlantic Treaty Organization (NATO)"),
        (87, "the World Bank"),
        (88, "the World Health Organization (WHO)"),
        (89, "the World Trade Organization (WTO)"),
    ]
    for qid, item in confidence_items:
        questions.append(
            _question(
                qid,
                "institutional_confidence",
                f"How much confidence do you have in {item}?",
                confidence_options,
            )
        )

    questions.append(
        _question(
            130,
            "migration",
            "What should the government do about people from other countries who want to come here to work?",
            [
                "Let anyone come who wants to.",
                "Let people come as long as there are jobs available.",
                "Place strict limits on the number of foreigners who can come here.",
                "Prohibit people coming here from other countries.",
            ],
        )
    )

    priority_options_1 = [
        "A high level of economic growth",
        "Making sure this country has strong defense forces",
        "Seeing that people have more say about how things are done at their jobs and in their communities",
        "Trying to make our cities and countryside more beautiful",
    ]
    questions.append(
        _question(
            152,
            "postmaterialist_index",
            "Looking ahead over the next ten years, which of the following goals should be the most important for this country?",
            priority_options_1,
        )
    )

    priority_options_2 = [
        "Maintaining order in the nation",
        "Giving people more say in important government decisions",
        "Fighting rising prices",
        "Protecting freedom of speech",
    ]
    questions.append(
        _question(
            154,
            "postmaterialist_index",
            "If you had to choose, which of the following should be most important for this country?",
            priority_options_2,
        )
    )

    priority_options_3 = [
        "A stable economy",
        "Progress toward a less impersonal and more humane society",
        "Progress toward a society in which ideas count more than money",
        "The fight against crime",
    ]
    questions.append(
        _question(
            156,
            "postmaterialist_index",
            "If you had to choose, which of the following should be most important for this country?",
            priority_options_3,
        )
    )

    surveillance_options = [
        "Definitely should have the right",
        "Probably should have the right",
        "Probably should not have the right",
        "Definitely should not have the right",
    ]
    for qid, item in [
        (196, "keep people under video surveillance in public areas"),
        (197, "monitor all emails and other information exchanged on the internet"),
        (198, "collect information about anyone living in this country without their knowledge"),
    ]:
        questions.append(
            _question(
                qid,
                "civil_liberties",
                f"Do you think your country's government should have the right to {item}?",
                surveillance_options,
            )
        )

    questions.append(
        _question(
            234,
            "political_culture",
            "How important is having honest elections for you?",
            importance_options,
        )
    )

    regime_options = ["Very good", "Fairly good", "Fairly bad", "Very bad"]
    for qid, item in [
        (235, "having a strong leader who does not have to bother with parliament and elections"),
        (236, "having experts, rather than government, make decisions according to what they think is best for the country"),
        (237, "having the army rule"),
        (238, "having a democratic political system"),
        (239, "having a system governed by religious law in which there are no political parties or elections"),
    ]:
        questions.append(
            _question(
                qid,
                "political_culture",
                f"How good or bad is the following as a way of governing this country: {item}?",
                regime_options,
            )
        )

    return sorted(questions, key=lambda row: row["question_id"])


def map_wvs_gender(value: object) -> str:
    try:
        value = int(value)
    except Exception:
        return "Prefer not to say"
    return {1: "Male", 2: "Female"}.get(value, "Prefer not to say")


def map_wvs_age(value: object) -> str:
    try:
        value = float(value)
    except Exception:
        return "Prefer not to say"
    if value < 0:
        return "Prefer not to say"
    if value <= 24:
        return "18-24 years old"
    if value <= 34:
        return "25-34 years old"
    if value <= 44:
        return "35-44 years old"
    if value <= 54:
        return "45-54 years old"
    if value <= 64:
        return "55-64 years old"
    return "65+ years old"


def map_wvs_education3(value: object) -> str:
    try:
        value = int(value)
    except Exception:
        return "Prefer not to say"
    if value < 0:
        return "Prefer not to say"
    if value <= 4:
        return "Secondary or less"
    if value == 5:
        return "Some tertiary"
    return "Bachelor or higher"


def map_wvs_religion(value: object) -> str:
    try:
        value = int(value)
    except Exception:
        return "Prefer not to say"
    if value < 0:
        return "Prefer not to say"
    if value == 0:
        return "No Affiliation"
    if value in {1, 2, 3}:
        return "Christian"
    if value == 4:
        return "Jewish"
    if value == 5:
        return "Muslim"
    return "Other"


def map_prism_education3(value: object) -> str:
    if value in {
        "Some Primary",
        "Completed Primary School",
        "Some Secondary",
        "Completed Secondary School",
        "Vocational",
    }:
        return "Secondary or less"
    if value == "Some University but no degree":
        return "Some tertiary"
    if value in {"University Bachelors Degree", "Graduate / Professional degree"}:
        return "Bachelor or higher"
    return "Prefer not to say"


def map_wvs_region(country_alpha: object) -> str:
    info = COUNTRY_INFO.get(str(country_alpha))
    return info["region"] if info else "Other"


def global_weighted_wvs_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["B_COUNTRY_ALPHA", "W_WEIGHT", "PWGHT", "Q260", "Q262", "Q275", "Q289"],
    )
    df["combined_weight"] = df["W_WEIGHT"].fillna(1.0) * df["PWGHT"].fillna(1.0)
    df["gender_global"] = df["Q260"].map(map_wvs_gender)
    df["age_global"] = df["Q262"].map(map_wvs_age)
    df["education3_global"] = df["Q275"].map(map_wvs_education3)
    df["religion_global"] = df["Q289"].map(map_wvs_religion)
    df["region_global"] = df["B_COUNTRY_ALPHA"].map(map_wvs_region)
    return df


def _normalized_weighted_distribution(df: pd.DataFrame, category_col: str) -> Dict[str, float]:
    grouped = df.groupby(category_col)["combined_weight"].sum()
    total = float(grouped.sum())
    if total <= 0:
        raise ValueError(f"No positive total weight for {category_col}")
    return {str(key): float(value / total) for key, value in grouped.items()}


def build_global_panel_config_dict(
    csv_path: Path,
    panel_size: int = 100,
    tolerance: float = 0.10,
) -> dict:
    df = global_weighted_wvs_frame(csv_path)
    gender_dist = _normalized_weighted_distribution(df, "gender_global")
    age_dist = _normalized_weighted_distribution(df, "age_global")
    region_dist = _normalized_weighted_distribution(df, "region_global")
    education_dist = _normalized_weighted_distribution(df, "education3_global")
    religion_dist = _normalized_weighted_distribution(df, "religion_global")
    return {
        "panel_size": panel_size,
        "tolerance": tolerance,
        "locale_filter": None,
        "attributes": [
            {
                "name": "region",
                "column": "location",
                "nested_key": "reside_region",
                "population_proportions": region_dist,
            },
            {
                "name": "gender",
                "column": "gender",
                "nested_key": None,
                "slack_categories": ["Prefer not to say"],
                "value_map": {"Non-binary / third gender": "Prefer not to say"},
                "population_proportions": gender_dist,
            },
            {
                "name": "age",
                "column": "age",
                "nested_key": None,
                "slack_categories": ["Prefer not to say"],
                "population_proportions": age_dist,
            },
            {
                "name": "education",
                "column": "education",
                "nested_key": None,
                "slack_categories": ["Prefer not to say"],
                "value_map": {
                    "Some Primary": "Secondary or less",
                    "Completed Primary School": "Secondary or less",
                    "Some Secondary": "Secondary or less",
                    "Completed Secondary School": "Secondary or less",
                    "Vocational": "Secondary or less",
                    "Some University but no degree": "Some tertiary",
                    "University Bachelors Degree": "Bachelor or higher",
                    "Graduate / Professional degree": "Bachelor or higher",
                },
                "population_proportions": education_dist,
            },
            {
                "name": "religion",
                "column": "religion",
                "nested_key": "simplified",
                "slack_categories": ["Prefer not to say"],
                "population_proportions": religion_dist,
            },
        ],
    }


def write_subjective_questions(output_path: Path) -> List[dict]:
    questions = build_subjective_questions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return questions


def write_global_panel_config(csv_path: Path, output_path: Path, panel_size: int, tolerance: float) -> dict:
    config = build_global_panel_config_dict(csv_path, panel_size=panel_size, tolerance=tolerance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def required_wvs_columns(questions: Iterable[dict]) -> List[str]:
    base_cols = ["B_COUNTRY_ALPHA", "W_WEIGHT", "PWGHT"]
    question_cols = [question["question_code"] for question in questions]
    return base_cols + sorted(set(question_cols))
