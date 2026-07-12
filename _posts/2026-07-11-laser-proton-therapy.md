---
layout: post
title: 'Laser-Driven Proton Therapy: Strong Physics, Hard Road to the Clinic'
date: 2026-07-11
permalink: /posts/2026/07/laser-proton-therapy/
description: "A follow-up to my post on VHEE electrons: why laser-driven proton acceleration is great physics but, in my view, unlikely to become a competitive medical product — and where its real strengths are."
tags:
  - Laser-Plasma Acceleration
  - Proton Therapy
  - FLASH Therapy
  - Medical Physics
  - TNSA
  - Radiotherapy
related_posts: true
---

### The Promise of a Small, Cheap Proton Source

Proton therapy has a real advantage over ordinary X-ray treatment. A proton beam deposits most of its energy at a chosen depth and then stops — a feature called the Bragg peak — so it can strike a tumor while sparing the healthy tissue before and behind it. The difficulty has always been the machine. Producing protons with enough energy for treatment, up to about 200 MeV to reach roughly 25 cm deep in human body, has traditionally required a cyclotron or synchrotron with hundred-ton magnets, a heavily shielded room, and, until recently, a budget in the hundreds of millions of dollars.

About twenty years ago, laser-driven ion acceleration seemed to offer a shortcut. When an intense (petawatt) laser pulse strikes a thin foil, it generates an enormous electric field on the rear side — on the order of a trillion volts per meter, tens of thousands of times stronger than the radio-frequency cavities of a conventional accelerator. In principle, such a field could push protons to treatment energy over several tens of microns rather than many meters. The promise was irresistible: a proton source small enough and cheap enough to bring a scarce treatment to far more patients.

It is a beautiful idea, and strong groups have pursued it for close to two decades. This post is the proton companion to my earlier one on [VHEE electrons]({{ '/posts/2025/08/laser-driven-radiotherapy/' | relative_url }}). The physics, in my opinion, has largely delivered — yet the case for laser-driven protons as a hospital product runs into trouble that has little to do with the physics at all. To be clear from the outset, this is not a criticism of the science, which is great and worth doing for its own sake. My concern is narrower, and it is only about one goal: turning the method into a hospital machine.

---

### The Physics Has (Almost) Delivered

The progress here is genuine and deserves acknowledgment. TNSA, the workhorse mechanism, routinely produces protons of several to tens MeV, and the maximum energy has climbed steadily. Recent experiments have reached nearly 150 MeV by chaining two acceleration stages at extreme laser intensity. New facilities are being built specifically to reach 100–200 MeV with usable beams at a selected energy, and the beams themselves keep improving — more reproducible, carrying more charge, with a narrower spread in energy.

Many of the problems once thought impossible have been solved. If the only question were whether a laser can make a proton beam good enough for therapy, the answer would be trending toward yes.

But that was never the hard question.

---

### The Gradient Was Never the Real Problem

A trillion-volt-per-meter field is spectacular, yet the field strength — the "gradient" — was never what made proton therapy machines large or expensive. A clinical system is far more than an accelerating gap. It is an instrument that must place a planned dose onto a three-dimensional tumor, safely and reproducibly, every day, at a price that competes. The laser's giant field solves the problem of making the acceleration part compact, but compact acceleration was never the part that made these machines big or costly in the first place.

The fair way to judge laser proton therapy, then, is against the tests that any real product must pass. Three stand out. Does ultra-fast, or "FLASH," delivery give the laser a unique medical advantage? Can its beam actually be shaped onto a whole tumor? And does the complete system cost less than what hospitals can already buy? On all three the answer, I think, is likely no — and the first is the most interesting, because it fails on physics grounds alone.

---

### Does FLASH Change the Picture? A Look from Basic Physics

The recent excitement around FLASH radiotherapy — the finding that delivering an entire dose in a fraction of a second appears to spare healthy tissue — looks tailor-made for lasers, which excel at delivering a very high dose rate in an extremely short burst. On closer inspection, though, it probably does not help, and the reason is worth working through.

The FLASH effect is defined by delivering the full dose, typically at least 4–8 Gy, at an average rate above about 40 Gy/s, with the whole exposure finished in under roughly 200 ms. The benefit appears to level off somewhere around 100–1000 Gy/s. What matters in that definition is an average rate and a total time; nothing in it refers to structure below a microsecond.

That points to the real question: can beam timing finer than a microsecond matter at all? There are only three ways the timing of a beam can reach the biology.

1. **Inside a single ion track.** The energy here is deposited in femtoseconds, but this happens for each particle as it passes, regardless of how the overall pulse is shaped. A single track already reaches a local dose rate of about 10¹² Gy/s, whether the beam is continuous or arrives in bunches. The laser's femtosecond pulse coincides with this timescale but cannot intensify it.
2. **Between tracks.** This is the only channel a beam can genuinely influence, and it works through chemistry. The reactive fragments left in water — hydroxyl radicals (·OH) and free electrons among them — live and react over about a nanosecond to a microsecond. A hydroxyl radical travels only some 3.6 nm in a nanosecond, roughly the size of one small cluster of fragments, and these clusters blur together within about a microsecond. The chemistry, in other words, runs on its own clock, from nanoseconds to microseconds. Deliver faster than that and the effect saturates; compressing the pulse from nanoseconds to picoseconds to femtoseconds likely changes nothing.
3. **Heating and other bulk effects.** These would require a real surge of energy, but 10 Gy is only 10 J/kg — a temperature rise of about two thousandths of a degree, no matter how quickly it arrives. There is no shock wave and no thermal effect.

The biology, then, has no clock faster than about a nanosecond for a beam to drive. The distinction that matters here is between two things that are easy to conflate. The *duration* of an individual pulse — femtoseconds versus nanoseconds — is wasted resolution, since nothing in the chemistry responds to it. The *dose delivered per pulse* is a different variable, and here a laser is genuinely unique: it could in principle place an entire treatment's worth of dose inside a single radical lifetime, a dose per chemistry window perhaps 10⁵ times what a cyclotron delivers, and no conventional machine can come close.

Whether that regime actually does anything is the real open question. If the FLASH effect is governed by transient oxygen depletion, which unfolds over milliseconds, then only the average rate and total time matter, a cyclotron already reaches them, and the laser's extreme dose-per-pulse buys nothing. If instead it is governed by radical–radical recombination, which depends on the instantaneous density of reactive fragments, then dose-per-pulse could matter and lasers would have a real, though narrow, opening. The evidence so far leans toward saturation once the mean rate clears roughly 40–100 Gy/s — rates conventional machines can meet — but the matter is not settled. My reading is that FLASH is unlikely to be the advantage that rescues laser proton therapy, while conceding it is the one place a laser-specific benefit cannot yet be ruled out.

---

### Delivery: A Pulsed Beam Is the Wrong Tool for Scanning

Even setting the biology aside, delivery poses a problem of its own. Modern proton therapy relies on pencil-beam scanning. The beam's energy sets its depth — the Bragg peak — while magnets sweep it from side to side, and the tumor is built up from perhaps 10–40 energy layers and thousands of individually weighted spots. This spot-by-spot painting is exactly what lets the dose conform to the tumor's shape, which is the whole point of using protons.

Scanning of this kind wants a beam that is nearly continuous, quick to change energy, and fast to steer — which is what a cyclotron provides. A laser source is different in kind: it delivers a large charge in a single short burst, today at only a few shots per second. No one seriously proposes rastering such a source spot by spot; the realistic plans use its broad energy spread and high per-shot charge to fill a volume at once, or capture and transport a large bunch for delivery in a few shots. The catch is that these routes rely on broad-field or "shoot-through" (single-energy) delivery, which gives up some of the conformity that motivated pencil-beam scanning to begin with. In fairness, the FLASH frontier is pushing conventional machines the same way: scanned FLASH is hard even for a cyclotron, because switching energy layer by layer is slow, so transmission delivery is a shared compromise rather than a laser-specific flaw.

That fairness cuts only so far. Shared compromise or not, the pulsed source still offers less conformity than modern scanned therapy, and its charge is limited within the therapeutic energy window once the beam has been energy-selected; higher repetition rates would help but remain distant at therapeutic energies. The very quality that makes a laser a superb physics source makes it an awkward treatment source.

---

### The Cost Problem

The original case rested on size and cost, but while the laser community advanced, so did the competition. Commercial single-room cyclotron systems now install for roughly \$30–40M, much reduced from the \$150M+ of the old multi-room centers. The decisive detail is that the accelerator itself accounts for only a fraction of that — very roughly half. The rest — the shielded room, the gantry, the imaging, the patient couch, the quality-control systems — does not shrink much when a laser replaces the cyclotron. Even a free and perfect laser source would cut the total cost only by around half.

And the laser is far from free. A single-shot multi-petawatt system runs on the order of $15–20M, and such costs are falling. A version firing at the repetition rate a clinic needs, with the improved pumping and cooling that implies, is a much larger undertaking, tens of millions and rising. It also brings its own shielding burden — the intense interaction sprays secondary electrons, neutrons, hard X-rays and an electromagnetic pulse — that is arguably worse per useful proton than a cyclotron's. Proponents point out that a single laser could in principle feed several treatment rooms, spreading its cost; that helps, but the per-room vault, gantry, imaging and target handling remain, and those are where much of the money goes. Add a target "factory" able to deliver fresh, ultrathin foils many times a second, reliably, for years, and the whole-system arithmetic does not favor the laser today.

---

### Outlook

Laser-driven ion acceleration will remain one of the most exciting tools in physics — and it will probably not become a competitive proton-therapy product in five to ten years. Someone will very likely treat a patient with a laser-driven proton beam, and that will be a genuine milestone worth celebrating. But demonstrating that something can work and persuading a hospital to buy the next one are entirely different achievements. The machine that treats the first patient will still, I suspect, struggle to win the following purchase against a cyclotron judged on reliability, dose shaping, and cost.

I would be glad to be proven wrong, and the routes are clear enough. A demonstrated way to produce a single-energy, high-charge beam — through radiation-pressure acceleration, sometimes called "light sail," or related schemes — that tames both the energy spread and the charge at a high repetition rate. A sharp fall in the cost of high-power lasers, carried along by the fusion-energy supply chain. Firm evidence that the FLASH benefit truly requires the ultrafast pulse only a laser can provide. Or a treatment paradigm built around the pulsed beam's own strengths, rather than forced to imitate a cyclotron — volumetric single-shot delivery, or spatially fractionated "minibeam" schemes — that turns the time structure from a liability into a feature. My argument judges the laser against how proton therapy is practiced today; a genuinely new way to deliver dose could rewrite the comparison.

The effort belongs where the laser genuinely excels: ultrafast radiation biology and chemistry at dose rates nothing else can reach, compact pulsed sources of neutrons and isotopes, and warm-dense-matter research. These are not consolation prizes, but frontiers where a femtosecond, high-charge burst is exactly the right instrument, and where laser-driven ion beams may prove genuinely transformative. Whatever caution belongs here is only about framing. A great scientific tool need not become a hospital machine to be worth building; it is already unmatched at what it does best.
