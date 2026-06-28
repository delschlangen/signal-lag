# Societal Context — current real-world state
#
# ILLUSTRATIVE AND NON-EXHAUSTIVE. This file enriches the foresight synthesis with
# CURRENT real-world state across societal domains. It is NOT a filter, a fixed list,
# or a set of "the risks that matter." The synthesis reasons across the FULL
# STEEP/PESTLE+ framework regardless of what is written here. The purpose of this file
# is solely to UPDATE the model on current real-world conditions it cannot otherwise
# know from training data — not to narrow, rank, or pre-select what it considers.
# The model must treat everything below as background state to reason ACROSS and
# combine with the week's research signals, never as the only factors in play.
#
# Write BROAD, FACTUAL, and CURRENT. State conditions of the world, not conclusions
# about which are dangerous. Give the board, not the move. Update regularly.
#
# Last updated: June 2026

## Social
- US public sentiment toward AI has soured over the past year. A March 2026 Quinnipiac poll found 55% of Americans think AI will do more harm than good in daily life (up from 44% in April 2025); 64% think AI will do more harm than good for education.
- A March 2026 NBC News poll found 57% of voters believe AI's risks outweigh its benefits (vs. 34% the reverse); only 26% have positive feelings about AI, 46% negative — a net rating lower than every tested topic except the Democratic Party and Iran.
- Trust is low and usage is rising simultaneously: a Feb 2026 Verasight poll found 56% remain anxious about AI even as adoption grows. The Stanford 2026 AI Index notes global "more benefits than drawbacks" sentiment rose to 59%, but nervousness also rose to 52%.
- Sentiment is sharply split by demographics (see Demographic). The single strongest predictor of attitude is personal usage: daily users are favorable by +57 points; rare/never users unfavorable by -42.
- 76% of Americans think businesses are not transparent enough about their AI use; trust in AI developers is a recurring weak point.

## Technological
- Frontier capability is advancing fastest in agentic/autonomous systems and coding. Labs report models authoring large shares of their own code; autonomous task-completion horizons are lengthening.
- Capability is increasingly announced first by labs (system cards, blog posts) ahead of the academic safety literature's response — the lead-time gap this tool measures.
- Synthetic media generation is mainstream (a quarter of 2026 Super Bowl ads were AI-made or AI-advertising); detection and provenance lag generation.
- Agentic deployment is shifting risk from single-output failures toward long-horizon, multi-step, multi-tool behavior — outpacing existing monitoring designed for bounded tasks.

## Economic
- Intense "AI bubble" debate. Total AI spending projected to surpass $1.6T; OpenAI committed ~$1.4T over 8 years on data centers; the big four (Google, Microsoft, Meta, Amazon) expected to spend ~$450B in capex in 2026; the 14 largest data-center operators near $750B in 2026 (up from ~$450B in 2025); Goldman estimates ~$7.6T cumulative AI CapEx 2026–2031.
- Concentration risk: in late 2025 the five largest companies held up ~30% of the S&P 500 — the greatest concentration in half a century. AI-related stocks have driven ~75% of S&P 500 returns, 80% of earnings growth, and 90% of capex growth since ChatGPT launched.
- Circular-investment concerns (Nvidia↔OpenAI↔AMD↔Microsoft↔CoreWeave cross-holdings) fuel bubble skepticism. Hyperscalers took on ~$121B in new debt (+300% vs. typical load), some structured off-balance-sheet via special-purpose vehicles.
- Revenue/ROI doubts: an MIT study found 95% of organizations saw zero return on GenAI investment; one analysis found only ~3% of people pay for AI. A Feb 2026 NBER study found 90% of firms reported no productivity/workplace impact.
- Institutional warnings: the Bank of England and IMF have warned of global market-correction risk from AI overvaluation. Counter-view (JPMorgan, Fed Chair Powell, White House AI czar David Sacks, Ben Horowitz) holds AI generates real revenue and is not a classic bubble. DeepSeek's Jan 2025 release triggered a ~$600B single-day Nvidia drop; the KOSPI halted trading on June 23, 2026.

## Environmental
- Energy is becoming a binding constraint on AI buildout. The IEA projects global data-center electricity use rising from ~415 TWh in 2024 (~1.5% of total) toward ~945 TWh by 2030 (roughly 4×).
- Data-center siting, water use, and grid strain are emerging as local political flashpoints. Public opinion favors data-center approaches that protect consumers from high electricity costs and environmental impact.
- Grid capacity, power procurement, and cooling are now first-order bottlenecks alongside chip supply.

## Political
- The federal posture is deregulatory and pro-preemption. A Dec 11, 2025 executive order ("Ensuring a National Policy Framework for AI," EO 14365) directs a DOJ litigation task force (active Jan 10, 2026) to challenge state AI laws, a Commerce review of "burdensome" state laws, and an FTC policy stance; it exempts child safety, AI compute/data-center infrastructure, and state procurement from preemption.
- In March 2026 the White House released a National Policy Framework urging Congress to enact broad preemption under a light-touch standard.
- AI is politically "up for grabs" — voters trust neither party to handle it well; negative sentiment concentrates among younger voters and women under 50.
- A notable administration reversal: driven by national-security concerns over advanced model capabilities, the administration is reportedly considering pre-release frontier-model evaluation requirements, and CAISI (the renamed US AI Safety Institute) announced evaluation partnerships with Google, Microsoft, and xAI — a shift for an administration that entered opposing AI oversight.

## Legal / Regulatory
- Preemption is NOT settled law. A 10-year state-law moratorium failed in the Senate 99-1 (July 2025). State laws remain enforceable unless/until preempted by statute or struck down by courts. Prudent posture is to keep complying with state law while monitoring.
- California SB 53 (Transparency in Frontier AI Act) is in effect as of Jan 1, 2026: frontier developers (>10^26 FLOPS) must publish risk frameworks, report safety incidents, and provide whistleblower protections; penalties up to $1M/violation for firms over $500M revenue. Also effective: AB 2013 (training-data transparency); SB 942 (AI-content disclosure/watermarking, delayed to Aug 2, 2026).
- Texas TRAIGA (HB 149) in force Jan 1, 2026 — scaled back from an EU-style draft to prohibiting specific harmful practices, intent-based liability, AG enforcement with a 60-day cure period.
- Colorado's comprehensive SB 24-205 never took effect; in May 2026 it was repealed/replaced by the narrower SB 26-189 (ADMT-focused), effective Jan 1, 2027. It was the only state law named in the Dec 2025 EO.
- New York amended the RAISE Act (March 2026) to mirror SB 53 (revenue-based threshold), easing multistate compliance.
- A bipartisan federal discussion draft — the Great American AI Act (June 4, 2026, Obernolte/Trahan, 269 pages) — proposes the first comprehensive federal framework in exchange for a 3-year freeze on state laws regulating how models are *built* (not how they're *used*). Not yet introduced or voted on.
- State legislative volume is high: ~1,561 AI bills introduced across 45 states as of March 2026, surpassing all of 2024.
- EU AI Act: GPAI obligations took effect Aug 2, 2025; high-risk obligations have an Aug 2, 2026 deadline, though the Commission proposed (Nov 2025) delaying high-risk implementation toward 2027. It remains the binding global benchmark.

## Security / Geopolitical
- State-linked misuse of frontier AI is documented and rising. In June 2026, OpenAI disclosed PRC-linked influence operations using ChatGPT to amplify US domestic grievances — notably targeting the data-center/electricity-cost debate — rated low-impact (Category One) but notable for testing narratives against AI infrastructure itself.
- US–China competition centers on compute: export controls on advanced chips and high-bandwidth memory; Chinese stockpiling ahead of restrictions; Chinese model developers (DeepSeek and others) demonstrating competitive capability at lower cost.
- Frontier cyber capability is operationalizing: documented cases of AI-assisted/AI-orchestrated cyber operations and the first AI-developed exploit code; labs now rate some models "High" in cyber capability.
- National-security concern over advanced model capabilities is reshaping US oversight posture (see Political) — government model-review arrangements with major labs are now in place.
- Critical-infrastructure exposure (power grid, financial systems) to AI-enabled attacks is a growing institutional concern.

## Demographic
- Attitudes split sharply by group. Data for Progress (Feb 2026): men favorable +16, women unfavorable -10; under-45 favorable +25, 45+ unfavorable -10; Black voters +29 and Latino +10 favorable, white voters -3.
- Generational pessimism is strongest among the young despite high usage: Quinnipiac found Gen Z 36% support / 58% oppose AI, the most negative of any generation; NBC found 18-34s at net -44 favorability and women 18-49 at -41.
- A usage/education divergence at work: among employed college-educated voters, daily AI use jumped from 22% (Aug 2025) to 34% (early 2026), while non-college use fell ~6 points — a widening adoption gap.
- Overall adoption keeps rising: ~56-66% report using an AI tool in recent months; professional/managerial and white-collar workers adopt far faster than blue-collar workers and retirees.
