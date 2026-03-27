# UK Panel Config Sources

This config is intended for PRISM's UK-locale pool and uses exact or directly derived
England and Wales Census 2021 distributions from official UK government sources.

## Gender

Source:
- ONS, *Population and household estimates, England and Wales: Census 2021*
- https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/populationandhouseholdestimatesenglandandwales/census2021

Values used:
- Female: 51.0%
- Male: 49.0%

PRISM mapping:
- `Non-binary / third gender` and `Prefer not to say` are retained as zero-target slack categories.

## Age

Source:
- GOV.UK Ethnicity facts and figures, *Age groups*
- https://www.ethnicity-facts-figures.service.gov.uk/uk-population-by-ethnicity/demographics/age-groups/latest/
- Underlying source on page: ONS Census 2021, England and Wales

Raw official age counts used:
- 18 to 24: 4,957,265
- 25 to 29: 3,901,735
- 30 to 34: 4,148,785
- 35 to 39: 3,981,630
- 40 to 44: 3,755,770
- 45 to 49: 3,788,730
- 50 to 54: 4,123,455
- 55 to 59: 4,029,040
- 60 to 64: 3,455,580
- 65 to 69: 2,945,140
- 70 to 74: 2,977,975
- 75 to 79: 2,170,265
- 80 to 84: 1,515,085
- 85 and over: 1,454,740

These were aggregated to PRISM's adult bins and renormalized over the 18+ population:
- 18-24 years old: 0.1050152425
- 25-34 years old: 0.1705430938
- 35-44 years old: 0.1639099256
- 45-54 years old: 0.1676125901
- 55-64 years old: 0.1585550065
- 65+ years old: 0.2343641415

## Ethnicity

Source:
- ONS, *Ethnic group, England and Wales: Census 2021*
- https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/ethnicity/bulletins/ethnicgroupenglandandwales/census2021

Values used:
- White: 81.7%
- Asian: 9.3%
- Mixed: 3.0%
- Other: 2.1%
- Black: 3.9%

Notes:
- The ONS bulletin reports the five high-level categories for England and Wales. The `Black` share is the residual to sum to 100% after the other four reported top-level categories.
- PRISM's `Hispanic` is mapped to `Other`, since the UK census high-level ethnicity schema has no separate Hispanic top-level category.
- `Prefer not to say` is kept as a zero-target slack category for compatibility with PRISM.

## Religion

Source:
- ONS, *Religion, England and Wales: Census 2021*
- https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/religion/bulletins/religionenglandandwales/census2021

Values used:
- Christian: 46.2%
- No religion / no affiliation: 37.2%
- Muslim: 6.5%
- Jewish: 0.5%
- Other: 3.6%
- Prefer not to say: 6.0%

Notes:
- `Other` is the sum of Hindu (1.7%), Sikh (0.9%), Buddhist (0.3%), and Other religion (0.7%).
- `Prefer not to say` is the residual nonresponse on the voluntary religion question, using the bulletin's reported 94.0% response rate.

## Education

Source:
- GOV.UK Explore Education Statistics, *Education and training statistics for the UK*
- https://explore-education-statistics.service.gov.uk/find-statistics/education-and-training-statistics-for-the-uk

Exact threshold shares used:
- NQF level 3 or above, adults aged 19 to 64: 66%
- NQF level 4 or above, adults aged 19 to 64: 47%

Derived coarse proportions used:
- Secondary or less: 34.0% (`100% - 66%`)
- Some tertiary: 19.0% (`66% - 47%`)
- Bachelor or higher: 47.0%

PRISM mapping:
- `Some Secondary`, `Completed Secondary School` -> `Secondary or less`
- `Vocational`, `Some University but no degree` -> `Some tertiary`
- `University Bachelors Degree`, `Graduate / Professional degree` -> `Bachelor or higher`
- `Prefer not to say` is retained as a zero-target slack category.

Notes:
- This is a coarse exact-source proxy rather than a perfect qualification-equivalence mapping. The source provides official threshold shares rather than the exact PRISM label schema.

## Employment status

Source:
- ONS, *Economic activity status, England and Wales: Census 2021*
- https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/economicactivitystatusenglandandwales/census2021

Exact official values used:
- In employment: 57.2%
- Of those employed, full-time: 70.2%
- Of those employed, part-time: 29.8%
- Unemployed: 3.4%
- Retired: 21.6%
- Student: 5.6%
- Looking after home or family: 4.8%
- Long-term sick or disabled: 4.2%
- Other economically inactive: 3.1%

Derived PRISM-aligned proportions used:
- Working full-time: 40.1946% (`57.2% * 70.2%`)
- Working part-time: 17.0627% (`57.2% * 29.8%`)
- Student: 5.6056%
- Unemployed, seeking work: 3.4034%
- Retired: 21.6216%
- Homemaker / Stay-at-home parent: 4.8048%
- Unemployed, not seeking work: 7.3073% (`long-term sick or disabled + other inactive`)

Notes:
- The raw official percentages sum to 99.9% because of source rounding, so the PRISM-aligned targets were renormalized to sum to 100%.
- `Prefer not to say` is retained as a zero-target slack category.

## Marital status

Source:
- ONS, *Marriage and civil partnership status, England and Wales: Census 2021*
- https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/marriagecohabitationandcivilpartnerships/articles/marriageandcivilpartnershipstatusenglandandwalescensus2021/2023-02-22

Exact official values used:
- Never married / never in a civil partnership: 37.9%
- Married or in a civil partnership: 44.6%
- Divorced or civil partnership dissolved: 9.1%
- Widowed or surviving partner: 6.1%
- Separated but still legally married / in a civil partnership: 2.2%

Derived PRISM-aligned proportions used:
- Never been married: 37.9%
- Married: 44.6%
- Previously married / separated / widowed: 17.5% in the config (from a raw 17.4% = `9.1% + 6.1% + 2.2%`, rounded so the stored targets sum to 100%)

PRISM mapping:
- `Divorced / Separated`, `Widowed` -> `Previously married / separated / widowed`
- `Prefer not to say` is retained as a zero-target slack category.

Notes:
- I use a combined prior-partnership bucket because the exact official categories split `divorced`, `widowed`, and `separated`, while the PRISM UK pool has only six `Widowed` respondents, which is too sparse to support a stable standalone widowed quota at `k=145`.
