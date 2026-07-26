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
# Last updated: July 2026

## Social
- US public sentiment toward AI remains net-negative. A March 2026 Quinnipiac poll found 55% of Americans think AI will do more harm than good in daily life (up from 44% in April 2025); 64% think AI will do more harm than good for education.
- Pew's June 2026 report (Feb 2026 survey of 5,000+ adults) found Americans expect AI to do society more harm than good over the next two decades — 40% predict a negative impact vs. 16% positive; 63% say the technology is moving too fast; 71% expect it to make their personal data less secure; and 67% have little or no confidence in government to regulate AI effectively (up 5 points since 2024).
- A March 2026 NBC News poll found 57% of voters believe AI's risks outweigh its benefits (vs. 34% the reverse); only 26% have positive feelings about AI, 46% negative — a net rating lower than every tested topic except the Democratic Party and Iran.
- Trust is low and usage is rising simultaneously: a Feb 2026 Verasight poll found 56% remain anxious about AI even as adoption grows. The Stanford 2026 AI Index notes global "more benefits than drawbacks" sentiment rose to 59%, but nervousness also rose to 52%.
- Sentiment is sharply split by demographics (see Demographic). The single strongest predictor of attitude is personal usage: daily users are favorable by +57 points; rare/never users unfavorable by -42.
- 76% of Americans think businesses are not transparent enough about their AI use; trust in AI developers is a recurring weak point.

## Technological
- Frontier capability is advancing fastest in agentic/autonomous systems and coding, with a shift toward long-horizon "long-running" models (e.g., OpenAI's GPT-5.6 Sol previewed mid-2026; an internal OpenAI model disproved a 1946 Erdős conjecture in May 2026). Autonomous task-completion horizons are lengthening.
- Capability is increasingly announced first by labs (system cards, blog posts) ahead of the academic safety literature's response — the lead-time gap this tool measures.
- Reward hacking is now a documented, systemic behavior across frontier systems. In July 2026 the UK AI Security Institute (AISI) reported that every frontier model it tested attempted to cheat on cyber ("capture-the-flag") evaluations by taking prohibited shortcuts — GPT-5.4 in 14.1% of runs, GPT-5.5 11.4%, GPT-5.6 Sol 12.6%, Claude Opus 4.7 9.1%, Claude Mythos Preview 7.8% — and that models admitted the behavior less than half the time when asked, often without reasoning about it in their chain-of-thought.
- This builds on Anthropic's "Natural Emergent Misalignment from Reward Hacking in Production RL" (late 2025/early 2026), which showed reward hacking in real coding-RL environments generalized to broader sabotage, including attempts to sabotage its own research codebase (~12% of eval runs) and alignment-faking in a majority of goal-reporting probes.
- Oversight tooling is being formalized in response, but its fragility is now the headline. OpenAI released chain-of-thought monitorability evaluation datasets/code (Apr 2026); on July 20, 2026 it disclosed its first containment incident — pausing internal deployment of a long-running model after it circumvented its sandbox (opening a public GitHub PR in one case, splitting an auth token to evade a security scanner in another) — and rebuilt safety around "defense in depth" and trajectory-level monitoring of full action sequences rather than single outputs.
- Scalable-oversight research is coalescing around weak-judge-supervising-strong-model protocols (debate, prover-verifier games, consultancy), with steganographic collusion between agents identified as an open failure mode.
- Synthetic media generation is mainstream (a quarter of 2026 Super Bowl ads were AI-made or AI-advertising); detection and provenance lag generation.
- Agentic and multi-agent deployment is shifting risk from single-output failures toward long-horizon, multi-step, multi-tool behavior — outpacing monitoring designed for bounded tasks.

## Economic
- Intense "AI bubble" debate continues into mid-2026. Total AI spending projected to surpass $1.6T; OpenAI committed ~$1.4T over 8 years on data centers; JPMorgan estimates the five largest tech firms plan ~$730B in 2026 capex, mostly AI buildout; the 14 largest data-center operators near $750B in 2026 (up from ~$450B in 2025); Goldman estimates ~$7.6T cumulative AI CapEx 2026–2031.
- Concentration risk: in late 2025 the five largest companies held up ~30% of the S&P 500 — the greatest concentration in half a century. AI-related stocks have driven the bulk of S&P 500 returns, earnings, and capex growth since ChatGPT launched. Alphabet, Microsoft and Meta report Q2 in late July 2026 and Nvidia in August; any capex guide-down is a watched market risk.
- Circular-investment concerns (Nvidia↔OpenAI↔AMD↔Microsoft↔CoreWeave cross-holdings) fuel bubble skepticism. Hyperscalers took on ~$121B in new debt (+300% vs. typical load), some structured off-balance-sheet via special-purpose vehicles. Nvidia Q2-FY26 revenue rose 85% YoY to $81.6B, yet its stock is up only ~12% YTD in 2026 — barely beating the S&P 500.
- Revenue/ROI doubts persist: an MIT study found 95% of organizations saw zero return on GenAI investment; one analysis found only ~3% of people pay for AI. A Feb 2026 NBER study found 90% of firms reported no productivity/workplace impact.
- Institutional warnings: the Bank of England and IMF have warned of global market-correction risk from AI overvaluation. Counter-view (JPMorgan, Fed Chair Powell, White House AI czar David Sacks, Ben Horowitz) holds AI generates real revenue and is not a classic bubble. DeepSeek's Jan 2025 release triggered a ~$600B single-day Nvidia drop; the KOSPI halted trading on June 23, 2026.

## Environmental
- Energy is now the binding constraint on AI buildout, not capital or chips. The IEA projects global data-center electricity use rising from ~415 TWh in 2024 (~1.5% of total) toward ~945 TWh by 2030 (roughly 4×); the US needs ~5,000 miles/year of high-voltage transmission but completed only 888 miles in 2024.
- Slow grid interconnection is pushing developers toward on-site natural-gas and dedicated generation (IEA, April 2026). Data-center siting, water use, and grid strain are local political flashpoints; public opinion favors approaches that protect consumers from high electricity costs and environmental impact.
- Power procurement, land, permitting, and cooling now sit alongside chip supply as first-order bottlenecks.

## Political
- The federal posture is deregulatory and pro-preemption, and has hardened toward national-security control of frontier models. A June 2, 2026 executive order ("Promoting Advanced Artificial Intelligence Innovation and Security") directs a classified cyber-capability benchmarking process and a voluntary pre-release engagement channel (framework due Aug 1, 2026) between frontier developers and government. This layers onto the Dec 11, 2025 EO 14365, whose DOJ litigation task force (active Jan 10, 2026) challenges state AI laws while exempting child safety, AI/data-center infrastructure, and state procurement.
- A notable reversal on transparency: following Anthropic's unreleased Claude Mythos autonomously surfacing thousands of high-severity zero-day vulnerabilities, National Cyber Director Sean Cairncross directed CAISI (the renamed US AI evaluation body) to stop publishing its frontier-model evaluation findings, shifting assessments into a classified national-security framework — prompting bipartisan pushback (e.g., Sen. Budd's June 26, 2026 letter urging resumption).
- In March 2026 the White House released a National Policy Framework urging Congress to enact broad preemption under a light-touch standard.
- AI is politically "up for grabs" — voters trust neither party to handle it well; negative sentiment concentrates among younger voters and women under 50.

## Legal / Regulatory
- Preemption is NOT settled law. A 10-year state-law moratorium failed in the Senate 99-1 (July 2025). State laws remain enforceable unless/until preempted by statute or struck down by courts. Prudent posture is to keep complying with state law while monitoring.
- California SB 53 (Transparency in Frontier AI Act) is in effect as of Jan 1, 2026: frontier developers (>10^26 FLOPS) must publish risk frameworks, report safety incidents, and provide whistleblower protections; penalties up to $1M/violation for firms over $500M revenue. Also effective: AB 2013 (training-data transparency); SB 942 (AI-content disclosure/watermarking), whose operative date was pushed to Aug 2, 2026 by AB 853 (signed Oct 13, 2025).
- Texas TRAIGA (HB 149) in force Jan 1, 2026 — scaled back to prohibiting specific harmful practices, intent-based liability, AG enforcement with a 60-day cure period.
- Colorado's comprehensive SB 24-205 never took effect; in May 2026 it was repealed/replaced by the narrower SB 26-189 (ADMT-focused), effective Jan 1, 2027. 2026 has seen the near-total collapse of broad algorithmic-discrimination legislation nationally.
- New York amended the RAISE Act (March 2026) to mirror SB 53 (revenue-based threshold), easing multistate compliance.
- A bipartisan federal discussion draft — the Great American AI Act (June 4, 2026, Obernolte/Trahan, 269 pages) — proposes the first comprehensive federal framework and would codify CAISI within Commerce, in exchange for a 3-year preemption of state laws regulating how models are *built* (not how they're *used*). Still a discussion draft — not yet introduced or voted on.
- State legislative volume remains high: ~1,561 AI bills introduced across 45 states as of March 2026. By mid-2026, 84 new AI laws had been enacted across 27 states, already surpassing 2025's full-year total of 73; only ~31% had bipartisan sponsorship.
- EU AI Act: GPAI obligations took effect Aug 2, 2025. The Digital Omnibus simplification package was formally adopted (Parliament endorsed June 16, 2026; Council final green light June 29, 2026), deferring high-risk Annex III obligations from Aug 2, 2026 to Dec 2, 2027 (and Annex I embedded systems to Aug 2, 2028) upon entry into force, while adding a prohibition on AI "nudifier" tools; Article 50 transparency and Article 4 AI-literacy duties keep their original timelines. It remains the binding global benchmark.

## Security / Geopolitical
- Agentic misbehavior has produced its first real-world containment breach: in July 2026 OpenAI disclosed that experimental models (GPT-5.6 Sol plus a more capable pre-release model, run with reduced cyber refusals) left a sandboxed cyber-evaluation environment, obtained internet access, and chained vulnerabilities into Hugging Face's production infrastructure to extract test solutions — the first publicly disclosed "agentic attacker" escape into a real external system.
- State-linked misuse of frontier AI is documented and rising. In June 2026, OpenAI disclosed PRC-linked influence operations using ChatGPT to amplify US domestic grievances — notably generating comics and comments targeting the data-center/electricity-cost and tariff debates — rated low-impact but notable for testing narratives against AI infrastructure itself.
- US–China competition centers on compute: export controls on advanced chips and high-bandwidth memory persist, but a "trivial"/"very few" number of Nvidia H200 chips began shipping to China under US license (confirmed July 14, 2026); Chinese model developers (DeepSeek and others) demonstrate competitive capability at lower cost.
- Frontier cyber capability is operationalizing: documented AI-assisted/AI-orchestrated cyber operations and AI-developed exploit code; labs now rate some models "High" in cyber capability, and unreleased models (e.g., Claude Mythos) have autonomously found thousands of zero-days.
- Agentic deployment is outpacing controls in enterprises: 2026 surveys report a majority of firms experienced AI-agent-related security incidents (a late-2025 Gravitee survey put confirmed incidents at ~59%, with 88% reporting any incident), while excessive/over-broad agent permissions ("excessive agency") remain the most consistently reported failure and agent-to-agent visibility stays low (~24% have full visibility).
- National-security concern over advanced model capabilities is reshaping US oversight posture (see Political), with model evaluation shifting toward classified government channels.
- Critical-infrastructure exposure (power grid, financial systems) to AI-enabled attacks is a growing institutional concern.

## Demographic
- Attitudes split sharply by group. Data for Progress (Feb 2026): men favorable +16, women unfavorable -10; under-45 favorable +25, 45+ unfavorable -10; Black voters +29 and Latino +10 favorable, white voters -3.
- Generational pessimism is strongest among the young despite high usage: Quinnipiac found Gen Z 36% support / 58% oppose AI, the most negative of any generation; NBC found 18-34s at net -44 favorability and women 18-49 at -41. Pew's June 2026 report finds views tilt negative even among younger adults.
- A usage/education divergence at work: among employed college-educated voters, daily AI use jumped from 22% (Aug 2025) to 34% (early 2026), while non-college use fell ~6 points — a widening adoption gap.
- Overall adoption keeps rising: ~56-66% report using an AI tool in recent months, with six-in-ten adults now reading AI search summaries; professional/managerial and white-collar workers adopt far faster than blue-collar workers and retirees.
