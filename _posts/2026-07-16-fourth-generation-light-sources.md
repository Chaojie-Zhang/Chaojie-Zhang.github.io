---
layout: post
title: 'The Fourth Generation: Where the World’s X-Ray Light Sources Stand in 2026'
date: 2026-07-16
permalink: /posts/2026/07/fourth-generation-light-sources/
description: 'A survey of the synchrotrons and free-electron lasers that produce the world’s brightest X-rays — what is running, what is being rebuilt, what is only proposed — and an assessment of where plasma acceleration fits.'
tags:
  - Particle Accelerators
  - Synchrotron Light Sources
  - Free-Electron Laser
  - X-ray Laser
  - Laser-Plasma Acceleration
  - Advanced Accelerator Concepts
related_posts: true
---

Most of what we know about the atomic structure of matter — the shape of a protein, the way a battery electrode degrades, how a catalyst grips a molecule — we learned by shining X-rays on it. The X-rays come from a few dozen machines around the world, and those machines are in the middle of the largest transition in their history.

Seven of them now run on a design that did not exist in usable form fifteen years ago, and four of the seven switched on in the last two years. Several more of the best-known facilities are shutting down, having their interiors torn out, and being rebuilt inside their own tunnels before the decade is out — the first of them went dark last summer. On the free-electron laser side, one machine has just reached a repetition rate nearly a thousand times higher than the previous generation, and another has, for the first time, made an X-ray laser lase inside a resonant cavity.

I wanted a clear picture of all of this in one place, so I assembled one. Everything below comes from public documents: commissioning papers, design reports, budget justifications, laboratory announcements. Where a project's schedule has moved, I say so, because in this field the schedule tends to be the news.

At the end I offer an assessment of where plasma acceleration fits. That part is opinion, and since plasma acceleration is what I work on, I have tried to rely on other people's evaluations rather than my own.

---

### What These Machines Are, and What "Brightness" Means

A synchrotron light source is a ring, a few hundred meters to a couple of kilometers around, in which electrons circulate at very nearly the speed of light. Magnets bend them, and any charged particle forced to turn radiates. At these energies the radiation comes out as X-rays, in a narrow forward cone, and it is piped down "beamlines" to experiments arranged around the ring like spokes. A large facility runs thirty to sixty of them at once, which is the ring's great advantage: many experiments, all day, every day.

The figure of merit is **brightness**: how many photons per second, emitted from how small a spot, into how narrow a cone, within how narrow a band of color. All four matter. You need many photons because most of them miss. You need a small spot and a narrow cone because that is what lets you focus the beam onto something tiny, and because it is what makes the light *coherent* — able to interfere with itself, which is what turns a shadow into an image with atomic detail.

Spot size multiplied by angular spread is a single quantity called **emittance**, and it belongs to the electron beam, not the light. The X-rays inherit it. So the entire enterprise of building a better light source reduces, to first approximation, to one question: how do you make the circulating electron beam smaller and less divergent?

There is a floor. Light itself has a minimum emittance set by diffraction — roughly the wavelength divided by 4π. A beam at that floor is called **diffraction-limited**, and its light is fully coherent. Getting close to it is the whole point of the current generation, which is why these machines are called diffraction-limited storage rings.

---

### One Equation Explains Most of the Landscape

Here is the scaling that organizes everything that follows. A ring's natural emittance goes as

$$
\varepsilon_x \propto \gamma^2 \theta^3
$$

where \\(\gamma\\) is the electron energy in units of its rest mass and \\(\theta\\) is the angle through which each individual bending magnet turns the beam.

The important term is the cube. Emittance grows because bending shakes the beam: each time an electron radiates a photon it recoils slightly, and if it is being bent hard, those recoils add up into transverse spread. Bend gently and you shake it less — very much less, because of that exponent.

So: instead of a few strong magnets, use many weak ones. Split each bend into seven pieces and each piece turns the beam through one-seventh the angle, which costs a factor of \\(7^3 = 343\\) in emittance. That idea, proposed in 1993 and called the **multi-bend achromat**, is why the last decade happened. The price is that everything else gets harder — the beam must be focused far more strongly, which means smaller magnets packed tighter, vacuum chambers narrowed from tens of millimeters to twenty or less, and an unforgiving sensitivity to alignment.

The cube has a second consequence, less often stated. The magnets must collectively bend the beam through a full circle, so \\(\theta\\) is fixed by how many magnets you can fit, which is fixed by circumference. **At a given beam energy, circumference is destiny:**

| Ring | Energy | Circumference | Natural emittance |
|---|---|---|---|
| ESRF-EBS (France) | 6 GeV | 844 m | ~135 pm·rad |
| APS-U (USA) | 6 GeV | 1104 m | 42 pm design, 33 pm measured |
| HEPS (China) | 6 GeV | 1360 m | 34.2 pm |
| PETRA-IV (Germany) | 6 GeV | 2304 m | 20 pm target |

Four machines at the same energy, ordered by size. (A picometer-radian is \\(10^{-12}\\) meter-radians; for scale, the diffraction limit at 10 keV — a typical hard X-ray — is about 10 pm·rad.) PETRA-IV expects to lead this list not because of a physics insight but because it occupies a 2.3-kilometer tunnel built decades ago for a particle collider.

That is worth holding onto. Which facility has the world's brightest hard X-rays is a question answered substantially by civil engineering choices made in a previous era, and the answer will change when someone else finishes a bigger tunnel.

---

### The Machines That Are Running

| Facility | Location | Energy | Circumference | Emittance | Users since |
|---|---|---|---|---|---|
| MAX IV | Lund, Sweden | 3 GeV | 528 m | ~330 pm | 2016 |
| ESRF-EBS | Grenoble, France | 6 GeV | 844 m | ~135 pm | 2020 |
| SIRIUS | Campinas, Brazil | 3 GeV | 518 m | ~250 pm | 2021 |
| NanoTerasu | Sendai, Japan | 3 GeV | 349 m | 1.14 nm | 2024 |
| APS-U | Argonne, USA | 6 GeV | 1104 m | 33 pm measured | 2024 |
| SLS 2.0 | Villigen, Switzerland | 2.7 GeV | 288 m | ~135 pm | 2025 |
| HEPS | Beijing, China | 6 GeV | 1360 m | 34.2 pm | 2026 |

**MAX IV** got there first, in 2016, and everything since is downstream of what it proved: that a multi-bend achromat ring can actually be built and operated. Its injector is worth a note of its own. A single 3 GeV linear accelerator serves as continuous top-up injector to two storage rings and, separately, as the driver for a short-pulse facility — with two different electron sources feeding it, one optimized for filling the rings and one for the short-pulse beam. That arrangement has since become a template.

**ESRF-EBS** was the first machine to rebuild an existing hard X-ray ring inside its own tunnel, and the first to use the "hybrid" variant of the multi-bend achromat that most later designs adopted. It is the reason the other upgrades believe they can work.

**APS-U** is the largest recent success in the American program. The ring went dark in April 2023, restarted in 2024, and was verified that August as the brightest storage ring in the world. It received final project approval in January 2026 at a total cost of $815 million, delivered on budget and ahead of schedule, with nine new beamlines. The [FY2027 Department of Energy budget justification](https://www.energy.gov/documents/fy-2027-basic-energy-sciences-budget-request) reports a measured horizontal emittance of 33 pm·rad; the design goal was 42.

**NanoTerasu** is the instructive outlier. It is a green-field machine in Sendai, only 349 meters around, with an emittance of 1.14 nanometer-radians — thirty times larger than APS-U. That is not a shortfall; it is a choice. NanoTerasu is aimed at soft and tender X-rays around 1–3 keV, where the diffraction limit is much less demanding, and it is positioned explicitly as the complement to SPring-8's hard X-rays. It reached stable operation at 400 milliamps in 2025, doubling its output. It has also signalled plans to extend its injector linac and add a soft X-ray free-electron laser later.

**SLS 2.0** achieved the fastest turnaround anyone has managed: ring replaced inside the existing building between October 2023 and December 2024, [beam back in January 2025](https://www.psi.ch/en/news/psi-stories/sls-2-0-how-to-start-up-a-particle-accelerator), first experiments in August 2025, regular operation in 2026. The rebuild raised the beam energy from 2.4 to 2.7 GeV and improved the emittance by a factor of 40; brightness rose by well over an order of magnitude from the ring alone, and by up to three at beamlines whose undulators were replaced too. One detail I find telling: its bending magnets are permanent magnets, not electromagnets, which means the ring's energy is now fixed for good. That is a deliberate trade of flexibility for stability and electricity, and more machines will make it.

**HEPS** is the first high-energy fourth-generation ring built from scratch rather than retrofitted, with 34.2 pm·rad emittance; it has been in trial operation since December 2025 and begins regular user operation this year.

---

### The Machines Being Built, Rebuilt, or Waiting

| Project | Location | Energy | Status |
|---|---|---|---|
| ALS-U | Berkeley, USA | 2 GeV | Construction approved Nov 2022; cost and schedule revised 2026; shutdown no earlier than Oct 2027, ~22 months; completion ~FY2030 |
| PETRA-IV | Hamburg, Germany | 6 GeV | 20 pm target; [German Science Council recommended funding Mar 2026](https://www.wissenschaftsrat.de/download/2026/3152-26.pdf); final approval anticipated within 2026 |
| Diamond-II | Oxfordshire, UK | 3.5 GeV | 18-month shutdown from Dec 2027; first users on flagship beamlines Jan 2030; complete Mar 2030 |
| SOLEIL-II | Saint-Aubin, France | 2.75 GeV | Entered construction |
| Elettra 2.0 | Trieste, Italy | 2.4 GeV | Dark since July 2025; user operation targeted Jan 2027 |
| ALBA-II | Barcelona, Spain | 3 GeV | Design |
| HALF | Hefei, China | 2.2 GeV | 480 m, 86 pm; accelerator installation from Mar 2026; completion Sept 2028 |
| Korea-4GSR | Ochang, Korea | 4 GeV | 800 m, ~60 pm; completion 2029 |
| SAPS | Guangdong, China | 3.5 GeV | Design |
| SPring-8-II | Hyōgo, Japan | 6 GeV | Construction funded from FY2024; shutdown summer 2027; user operation resumes FY2029 |
| NSLS-II-U | Upton, USA | 3 GeV | Concept; not yet approved |
| SSRL-X | Menlo Park, USA | — | Published lattice studies; not yet approved |

Two features of this table deserve comment.

**Almost every European and American entry is a rebuild.** Diamond, SOLEIL, Elettra, ALBA, ALS, PETRA and SPring-8 are each taking their ring out and putting a new one in — Elettra went dark in July 2025, the first of the wave, and [SPring-8 follows in 2027](https://www.riken.jp/en/news_pubs/research_news/rr/20260317_1/) — and each rebuild costs its user community an 18-to-22-month blackout. On present schedules those blackouts overlap heavily in the late 2020s. This is a coordination problem the community has only begun to discuss in public, and it means the facilities that stay lit through the transition will matter more than their brightness rankings suggest.

**The schedules move in one direction.** ALS-U is the clearest case. Construction was approved in November 2022 against a $590 million baseline. Since then [its shutdown date has moved](https://als.lbl.gov/joint-als-als-u-statement-on-dark-time-delay/) from October 2025 to June 2026 to no earlier than October 2027, and project completion is now expected around FY2030, with a [revised cost and schedule approved in 2026](https://research.lbl.gov/2026/02/18/looking-forward-to-the-upgraded-als/). The hardware is real — the accumulator ring's magnet assemblies are mechanically installed and storage ring magnet production is authorized. What slipped was the plan. Any published light source schedule should be read with that in mind.

The last two rows are the American projects without approvals, and they illustrate how the U.S. system works. Large Department of Energy projects pass through a series of gates: first an approval that a capability gap exists at all, then a choice of technology and a cost range, then a frozen baseline, then permission to build, then permission to operate. The first gate — approving the *need*, not the design — costs nothing to sign and commits a future budget line, which makes it the hardest one to get when budgets are contracting.

**NSLS-II-U** was assessed as "absolutely central" by [an advisory subcommittee in May 2024](https://science.osti.gov/-/media/bes/besac/pdf/Reports/Report-to-BESAC-on-New-and-Upgraded-National-User-Facilities-2024-05-28Final.pdf): it would be the world's brightest source between 1 and 10 keV, ten to twenty times better than anything in the United States today, using an unusual magnet design that replaces long electromagnets with strings of permanent magnets and thereby cuts the ring's electricity consumption substantially. The same report asked for more engineering study and recommended bringing in the teams from other recent upgrades before the design is frozen. It has not passed the first gate. **SSRL-X** exists as [published lattice studies](https://arxiv.org/abs/2311.13667), including one option that would reuse a decommissioned collider tunnel — which is, per the equation above, the PETRA-IV move. It has not passed the first gate either.

---

### Free-Electron Lasers: The Other Half of the Field

A storage ring's electrons radiate independently, like a crowd of people each striking a bell at random. The light adds up in intensity but not in phase.

A free-electron laser does something different. Send a very high-quality electron beam down a long straight line of alternating magnets — an **undulator** — and the light the electrons emit reaches back and pushes on the electrons themselves, sorting them into thin sheets spaced one wavelength apart. Once sorted, they radiate in step, and the power grows exponentially rather than linearly. The result is a pulse perhaps a billion times brighter at its peak than anything a ring can produce, and a few femtoseconds long instead of tens of picoseconds.

The trade is arithmetic. A ring serves fifty beamlines simultaneously. An X-ray FEL serves two or three, and the older ones fire only around a hundred times a second. Peak brightness is spectacular; the *average* number of photons delivered to the world per year has, until recently, been comparable.

| Facility | Location | Energy | Technology | Status |
|---|---|---|---|---|
| LCLS | SLAC, USA | up to 15 GeV | copper linac, 120 Hz | Operating |
| LCLS-II | SLAC, USA | 4 GeV | superconducting, continuous | First light 2023; 93 kHz reached Dec 2025; 1 MHz target |
| LCLS-II-HE | SLAC, USA | 8 GeV | superconducting | Approved Sep 2024, $716M; completion expected FY2028 |
| European XFEL | Hamburg, Germany | 17.5 GeV | superconducting, burst mode | Operating; 3 undulator lines |
| SACLA | Hyōgo, Japan | 8 GeV | copper linac | Operating |
| SwissFEL | Villigen, Switzerland | 5.8 GeV | copper linac | Operating |
| PAL-XFEL | Pohang, Korea | 10 GeV | copper linac | Operating |
| FLASH / FERMI | Hamburg / Trieste | soft X-ray | superconducting / seeded | Operating |
| SXFEL | Shanghai, China | 1.5 GeV | copper linac | Operating |
| SHINE | Shanghai, China | 8 GeV | superconducting, continuous | 3.1 km; 0.4–25 keV at 1 MHz; first beam targeted 2026 |
| DCLS | Dalian, China | vacuum UV | — | Operating |

That arithmetic is what changed. A copper accelerator can only fire in short pulses, because it would melt otherwise. A superconducting one can run continuously, and that lifts the repetition rate from 120 per second to a million. **LCLS-II** [reached 93 kilohertz in December 2025](https://www6.slac.stanford.edu/news/2025-12-09-new-world-record-lcls-approaches-100000-pulses-second-path-million) on the way to a megahertz. **LCLS-II-HE** will double its energy to 8 GeV, extending megahertz operation to hard X-rays of 5 to 13 keV; it was [approved in September 2024](https://www6.slac.stanford.edu/news/2024-09-27-new-upgrade-will-supercharge-atomic-vision-worlds-most-powerful-x-ray-laser) at $716 million and should finish in FY2028. **SHINE**, in Shanghai, is an 8 GeV continuous-wave machine in 3.1 kilometers of tunnel 29 meters underground, aiming at its first electron beam this year.

When both are running there will be two continuous megahertz X-ray lasers in the world, and the average-flux comparison with storage rings will no longer be close.

---

### An X-Ray Cavity, Finally

An ordinary FEL starts from nothing — the electrons' own random noise, amplified. That works, but it means the output is a different jagged spectrum every shot: broad, spiky, and fluctuating by roughly 100 percent divided by the square root of the number of independent spikes.

The fix is the one every laser uses: put the light in a cavity and let it go around many times, so each pass is seeded by the last. For X-rays this is hard, because ordinary mirrors do not reflect them. Diamond crystals do, by Bragg diffraction, but only within an extremely narrow band of colors — roughly one part in \\(10^5\\), compared with the one part in \\(10^3\\) that an ordinary FEL produces.

That narrowness is the whole trick, and it explains a number that is often misread. The case for the proposed LCLS-X quotes a 100-to-1000-fold gain in *average spectral brightness* from cavity-based sources. That is not more photons. An X-ray oscillator's gain per pass is low — tens of percent — and its peak power is *below* that of a conventional FEL. The entire factor comes from the denominator: brightness is quoted per 0.1 percent of bandwidth, and the cavity cuts the bandwidth by a hundred to a thousand. What you actually buy is coherence along the pulse and a collapse of the shot-to-shot fluctuation from 100 percent to a few percent. For many experiments the stability is worth more than the brightness.

The catch is geometric and unforgiving: the light's round trip through the cavity must take exactly as long as the gap between electron bunches. That requires bunches arriving at megahertz rates, which requires a superconducting accelerator, which is why this concept sits behind LCLS-II-HE and SHINE in every plan rather than beside them.

It is no longer hypothetical. In January 2026, a team reported lasing from a diamond cavity 132.8 meters around at the European XFEL, at 6.952 keV, matched to that machine's 2.23 MHz bunch spacing, with the light building up bunch by bunch ([*Nature* **650**, 93](https://doi.org/10.1038/s41586-025-10025-x)). The optics were proven at SLAC first: a SLAC–Argonne collaboration stored hard X-ray pulses through dozens of round trips in a 14-meter diamond cavity at LCLS ([*Nature Photonics* **17**, 878](https://doi.org/10.1038/s41566-023-01267-0), 2023), and the same team's cavity-FEL project with RIKEN aims to demonstrate two-pass gain at 9.831 keV ([*Phys. Rev. Accel. Beams* **27**, 110701](https://doi.org/10.1103/PhysRevAccelBeams.27.110701)) using a workaround for their copper accelerator: two bunches, 218 nanoseconds apart.

In the space of a few years, the X-ray cavity went from a design study to a measurement.

---

### The Axis Nobody Has Divided Up

The usual summary of the storage ring landscape is coverage: one champion for soft X-rays, one for the middle, one for hard, spectrum complete.

That summary describes one axis — average brightness against photon energy — and the table near the top of this article shows that even that axis is a lease rather than a deed, since it goes to whoever finishes the largest tunnel next.

The more interesting point is that it is only one axis. Consider time. A storage ring's bunches are 10 to 100 picoseconds long, and nothing within the concept changes that: the bunch length is set by an equilibrium between the radiation the electrons emit and the radio-frequency field that restores their energy. Every workaround charges a toll. Slicing a short piece out of a long bunch with a laser costs almost all the flux. Running one isolated high-current bunch among many costs repetition rate. Deflecting cavities cost something else again.

The result is that the window between roughly one picosecond and one hundred femtoseconds belongs to nobody. FELs are faster; rings are slower; no amount of emittance reduction reaches it. If I had to name the largest capability gap in the light source landscape, I would name that rather than photon energy.

Several other axes look similarly thin, and none of them needs a tunnel. Electron sources are the shared upstream bottleneck for rings and FELs alike — the U.S. accelerator research program says in its own words that it will emphasize "significant improvements in very high brightness and high current electron sources." Detectors and data handling are needed by every project and owned by none; advisory reports keep saying that new detectors are required before existing machines can deliver what they promise. And automation is quietly the fastest-moving item: the headline achievement in the FY2027 U.S. budget document is a machine-learning system that tunes an FEL's electron beam, improving emittance by a factor of two and doing it ten times faster than an operator.

---

### Where Plasma Acceleration Fits

This is my own field, so I will lean on other people's assessments.

**The idea.** A conventional accelerator pushes electrons with radio waves inside metal cavities, and it is limited by electrical breakdown: push past roughly 30 to 100 million volts per meter and the metal arcs. A plasma cannot break down, because it is already broken down — it is a gas whose electrons have been stripped off. Drive a laser pulse or a dense electron bunch through it and the plasma electrons are shoved aside and snap back, forming a wave that trails the driver like a boat's wake. That wave holds electric fields of tens to a hundred *billion* volts per meter, a thousand times what metal allows. The same energy gain, in a thousandth of the length.

**What has actually been demonstrated.** Free-electron lasing driven by plasma-accelerated beams has now been shown four ways: [laser-driven at 27 nanometers](https://doi.org/10.1038/s41586-021-03678-x) by a group in China in 2021; [beam-driven at Frascati](https://doi.org/10.1038/s41586-022-04589-1) in 2022; [laser-driven and seeded](https://doi.org/10.1038/s41566-022-01104-w) by a French-German collaboration in 2023; and [laser-driven at Berkeley](https://newscenter.lbl.gov/2025/07/29/researchers-make-key-gains-in-unlocking-the-promise-of-compact-x-ray-free-electron-lasers/) in 2025. All four are below 1 GeV, and all four are at ultraviolet or longer wavelengths. None is an X-ray FEL. The gap between what has been demonstrated and what a light source needs is the central fact of this subject, and it is a large gap.

The beam physics has moved quickly. [Preserving the beam's emittance](https://doi.org/10.1038/s41467-024-50320-1) through a plasma stage has been demonstrated. So has preserving its energy spread below one percent, and actively compressing it afterward. So has extracting more energy from the wave than the driver put in per unit charge — a ratio called the transformer ratio, whose classical ceiling of 2 for a symmetric driver has been [exceeded in plasma](https://doi.org/10.1103/PhysRevLett.121.064801) by shaping the driver's current profile. And in April 2026 a laser-plasma-driven FEL ran continuously for more than eight hours ([*Phys. Rev. Accel. Beams* **29**, 041301](https://doi.org/10.1103/z2d3-bhyt)).

That last result addresses the objection that actually matters. Gradient was never the problem. A user facility is judged on uptime and on mean time between failures; a plasma source that performs beautifully on a good afternoon is not yet a component of one.

**Where it is being built.** The most advanced effort is European. EuPRAXIA is the first plasma accelerator project placed on the [European roadmap for research infrastructure](https://roadmap2021.esfri.eu/projects-and-landmarks/browse-the-catalogue/eupraxia/), and its beam-driven half at Frascati has [roughly €120 million committed](https://www.eupraxia-project.eu/major-boost-to-european-plasma-accelerator-facility.html) from the Italian government, the Latium region, and INFN. Its architecture is deliberately modest: a conventional photoinjector and X-band linac deliver up to 500 MeV at up to 100 pulses per second, a plasma stage boosts that to 1 GeV, and the result drives an FEL. First pilot users are targeted for 2028. The second site, laser-driven and aiming at 1 to 5 GeV, has been chosen: [ELI Beamlines](https://www.eli-laser.eu/news/eupraxia-selects-eli-for-second-laser-driven-accelerator-site/), outside Prague.

Elsewhere, DESY has published [a conceptual design](https://bib-pubdb1.desy.de/record/615183/files/PIP4_CDR.pdf) for a laser-plasma injector that would deliver 6 GeV bunches to top up PETRA-IV, driven by the petawatt upgrade of its KALDERA laser, noting that such an injector could eventually replace the conventional chain. KIT is building a storage ring, cSTART, [designed to be fed by a laser-plasma injector](https://publikationen.bibliothek.kit.edu/1000173412). The UK's CLARA accelerator reached its design energy of 250 MeV in 2025, with plasma-wakefield experiments — a plasma-driven FEL demonstration among the stated aims — in its exploitation program.

**The official assessment.** The [May 2024 advisory report](https://science.osti.gov/-/media/bes/besac/pdf/Reports/Report-to-BESAC-on-New-and-Upgraded-National-User-Facilities-2024-05-28Final.pdf) that evaluated candidate technologies for a future U.S. light source described plasma acceleration as potentially a thousand times more efficient than current linear accelerators, noted that its development is rapid — parenthetically, "including in China" — and concluded that in 10 to 15 years it may be mature enough for such a facility. I do not disagree with that. Its practical translation is precise: as a construction project this decade, essentially zero; as research aimed at the 2040s, real.

A quieter signal points the same way from a different direction. The FY2027 U.S. budget document lists, among the year's achievements in basic energy sciences, a compact free-electron laser result in which a national laboratory and an American company produced high-brightness electron beams from a plasma and achieved nearly a thousandfold amplification through an undulator — and the document itself observes that compact plasma-based FELs offer opportunity for scientific discovery and for industrial applications including microelectronics fabrication. The commercial side has moved accordingly: a Palo Alto company founded in 2021 is developing an accelerator-driven FEL to replace the tin-droplet plasma sources used in extreme-ultraviolet lithography, with one machine intended to feed up to twenty chip-printing tools. It [raised a Series B in July 2025](https://www.xlight.com/news), received a [CHIPS research letter of intent](https://www.nist.gov/news-events/news/2025/12/department-commerce-and-nist-announce-chips-research-and-development-letter) from the Commerce Department in December 2025, and plans a prototype in Albany from 2028. I have written about that direction [before](https://chaojiezhang.me/posts/2025/08/accelerators-moores-law/). For high-average-power FELs, the pull may currently be industrial rather than scientific.

**My reading.** Plasma acceleration will not replace a storage ring. A ring's value is fifty beamlines running at once at high average flux, and nothing about a plasma stage helps with that. It will not replace a superconducting FEL linac in this decade either; LCLS-II-HE and SHINE will define that frontier and cavity-based sources will extend it.

What it offers is narrower and worth stating plainly. It offers energy per meter, which matters only where the alternative is a tunnel that cannot be afforded or sited. It offers a beam whose quality is set by how it is born inside the plasma rather than inherited from whatever drove the wave, which matters where the available driver is not good enough on its own. And it offers energy that can be tuned stage by stage, which a fixed-length linac cannot do at all.

Those are real. Whether they are worth their cost depends on what the alternative costs, and the alternative — superconducting linacs, better electron guns, better undulators, better detectors — is getting cheaper and more reliable every year.

---

### Outlook

The picture in one paragraph. The fourth generation of storage rings has arrived and is spreading, mostly by rebuilding existing machines inside their own tunnels at a cost of roughly two dark years each; the title of brightest hard X-ray source will pass to whoever finishes the largest tunnel, and will keep passing; the era of continuous, megahertz-class X-ray lasers is opening in two places; X-ray cavities stopped being hypothetical in January; and plasma acceleration is a decade or more from a user facility.

Those last two clauses are not in tension. A technology fifteen years from usefulness can be entirely worth the intervening fifteen years, and this one has produced a great deal of physics along the way that had nothing to do with light sources.

I would be glad to be shown wrong about the timescale, and the routes are legible. A plasma-driven FEL at soft X-ray rather than ultraviolet wavelengths. A plasma source that meets a facility's availability requirement rather than a good day's. A high-transformer-ratio measurement at several GeV rather than at tens of MeV. Each of these is somebody's next experiment. If they all land, 10 to 15 years is conservative; if none do, it was generous.

---

### Sources

*All figures are from the public record: commissioning papers, design reports, advisory-committee and budget documents, and laboratory announcements. Schedules are as published, and corrections are welcome. The load-bearing sources:*

**Reports and budget documents**

- [Report to BESAC on New and Upgraded National User Facilities](https://science.osti.gov/-/media/bes/besac/pdf/Reports/Report-to-BESAC-on-New-and-Upgraded-National-User-Facilities-2024-05-28Final.pdf) — Basic Energy Sciences Advisory Committee subcommittee, May 2024. The NSLS-II-U assessment and the plasma-technology outlook quoted above.
- [FY 2027 Congressional Budget Justification, DOE Basic Energy Sciences](https://www.energy.gov/documents/fy-2027-basic-energy-sciences-budget-request) — APS-U completion and measured emittance, LCLS-II-HE and ALS-U status, the machine-learning tuning result, and the compact plasma-FEL highlight.
- [Wissenschaftsrat statement on PETRA IV](https://www.wissenschaftsrat.de/download/2026/3152-26.pdf) — German Science Council, March 2026.

**Storage rings**

- MAX IV — [the linac and its two guns](https://www.maxiv.lu.se/beamlines-accelerators/accelerators/guns-and-linear-accelerator) (MAX IV Laboratory).
- SLS 2.0 — [Willmott, *Synchrotron Radiation News* (2024)](https://doi.org/10.1080/08940886.2024.2312059); [PSI, "SLS 2.0: how to start up a particle accelerator" (2025)](https://www.psi.ch/en/news/psi-stories/sls-2-0-how-to-start-up-a-particle-accelerator).
- ALS-U — [joint statement on the dark-time delay (2023)](https://als.lbl.gov/joint-als-als-u-statement-on-dark-time-delay/); ["Looking forward to the upgraded ALS" (February 2026)](https://research.lbl.gov/2026/02/18/looking-forward-to-the-upgraded-als/).
- NanoTerasu — [accelerator commissioning, *Phys. Rev. Accel. Beams* **28**, 020701 (2025)](https://doi.org/10.1103/PhysRevAccelBeams.28.020701).
- HEPS — [design and commissioning overview, *AAPPS Bulletin* (2026)](https://doi.org/10.1007/s43673-026-00184-y).
- SPring-8-II — [FY2024 annual report chapter (PDF)](https://www.spring8.or.jp/pdf/en/ann_rep/24/SPring-8-II.pdf); [RIKEN research news, March 2026](https://www.riken.jp/en/news_pubs/research_news/rr/20260317_1/).
- Diamond-II — [project overview talk](https://indico.global/event/5643/).
- SOLEIL II — ["Entrance in the Construction Phase" (IPAC proceedings)](https://meow.elettra.eu/81/pdf/MOPB074.pdf).
- SSRL-X — [lattice-options study, arXiv:2311.13667](https://arxiv.org/abs/2311.13667).

**Free-electron lasers and cavities**

- LCLS-II at 93 kHz — [SLAC news, December 9, 2025](https://www6.slac.stanford.edu/news/2025-12-09-new-world-record-lcls-approaches-100000-pulses-second-path-million).
- LCLS-II-HE approval — [SLAC news, September 27, 2024](https://www6.slac.stanford.edu/news/2024-09-27-new-upgrade-will-supercharge-atomic-vision-worlds-most-powerful-x-ray-laser).
- SHINE — [Sixth Tone](https://www.sixthtone.com/news/1018387).
- Cavity lasing at the European XFEL — [*Nature* **650**, 93 (2026)](https://doi.org/10.1038/s41586-025-10025-x).
- X-ray cavity storage at LCLS — [Margraf et al., *Nature Photonics* **17**, 878 (2023)](https://doi.org/10.1038/s41566-023-01267-0).
- The cavity-based FEL project — [*Phys. Rev. Accel. Beams* **27**, 110701 (2024)](https://doi.org/10.1103/PhysRevAccelBeams.27.110701).
- X-ray oscillator output properties — [arXiv:1903.09317](https://arxiv.org/abs/1903.09317).

**Plasma acceleration**

- Wang et al., [*Nature* **595**, 516 (2021)](https://doi.org/10.1038/s41586-021-03678-x) — laser-driven lasing at 27 nm.
- Pompili et al., [*Nature* **605**, 659 (2022)](https://doi.org/10.1038/s41586-022-04589-1) — beam-driven plasma FEL.
- Labat et al., [*Nature Photonics* **17**, 150 (2023)](https://doi.org/10.1038/s41566-022-01104-w) — seeded laser-plasma FEL.
- BELLA lasing — [Berkeley Lab news, July 2025](https://newscenter.lbl.gov/2025/07/29/researchers-make-key-gains-in-unlocking-the-promise-of-compact-x-ray-free-electron-lasers/).
- Eight hours of continuous operation — [*Phys. Rev. Accel. Beams* **29**, 041301 (2026)](https://doi.org/10.1103/z2d3-bhyt).
- Emittance preservation — [*Nature Communications* **15** (2024)](https://doi.org/10.1038/s41467-024-50320-1).
- Transformer ratio above 2 — [Loisch et al., *Phys. Rev. Lett.* **121**, 064801 (2018)](https://doi.org/10.1103/PhysRevLett.121.064801).
- EuPRAXIA — [ESFRI Roadmap 2021](https://roadmap2021.esfri.eu/projects-and-landmarks/browse-the-catalogue/eupraxia/); [funding announcement](https://www.eupraxia-project.eu/major-boost-to-european-plasma-accelerator-facility.html); [ELI Beamlines chosen as second site](https://www.eli-laser.eu/news/eupraxia-selects-eli-for-second-laser-driven-accelerator-site/).
- Laser-plasma injector for PETRA IV — [conceptual design report (DESY)](https://bib-pubdb1.desy.de/record/615183/files/PIP4_CDR.pdf).
- cSTART — [laser-plasma injector for an electron storage ring (KIT)](https://publikationen.bibliothek.kit.edu/1000173412).
- xLight — [company news](https://www.xlight.com/news); [NIST CHIPS letter of intent, December 2025](https://www.nist.gov/news-events/news/2025/12/department-commerce-and-nist-announce-chips-research-and-development-letter).
