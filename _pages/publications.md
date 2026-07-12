---
layout: page
permalink: /publications/
title: publications
description: peer-reviewed publications in reverse chronological order
nav: true
nav_order: 1
pub_numbering: true
redirect_from:
  - /publications/2013-prl-high-brightness/
  - /publications/2014-prab-two-color-injection/
  - /publications/2014-prl-phase-space-dynamics/
  - /publications/2015-cpc-high-quality-beams/
  - /publications/2016-ppcf-colliding-injection/
  - /publications/2016-ppcf-low-energy-spread/
  - /publications/2016-prab-temporal-characterization/
  - /publications/2016-prl-nanoscale-bunching/
  - /publications/2016-prl-phase-space-matching/
  - /publications/2016-prl-transverse-instability/
  - /publications/2016-scientific-reports-wakefield/
  - /publications/2017-prl-wakefield/
  - /publications/2018-nat-photon-infrared-pulses/
  - /publications/2018-plasma-physics-control-fusion-wakes/
  - /publications/2018-pop-tri-stage-proton/
  - /publications/2018-ppcf-emittance-measurement/
  - /publications/2018-ppcf-phase-locked-rings/
  - /publications/2018-ppcf-phase-space-diagnostics/
  - /publications/2018-ppcf-probing-wakefields/
  - /publications/2018-pre-electron-ion-instability/
  - /publications/2019-pra-hollow-channel-dechirper/
  - /publications/2019-prab-down-ramp-injection/
  - /publications/2019-prl-plasma-dechirper/
  - /publications/2019-sci-rep-xray-imaging/
  - /publications/2019-science-advances/
  - /publications/2020-commsphys-angular-momentum-conservation/
  - /publications/2020-natcomm-photon-deceleration/
  - /publications/2020-njp-two-frequency-laser-tweezer/
  - /publications/2020-ppcf-anisotropic-electron-distribution/
  - /publications/2020-ppcf-weibel-instability-probing/
  - /publications/2020-prl-weibel/
  - /publications/2021-natphys-high-throughput-injection/
  - /publications/2021-plasma-physics-control-fusion-grating/
  - /publications/2021-pop-ultrashort-pulse-generation/
  - /publications/2021-prl-dense-gamma-ray-pulses/
  - /publications/2022-oe-cross-polarized-interferometry/
  - /publications/2022-pnas-weibel/
  - /publications/2022-pop-electron-weibel-instability/
  - /publications/2022-prresearch-spin-polarized-electron-beams/
  - /publications/2023-prab-underdense-plasma-lens/
  - /publications/2023-prresearch-tunable-vortices-plasmas/
  - /publications/2023-review-modern-plasma-physics/
  - /publications/2024-commsphys-vortex-vector-laser-pulses/
  - /publications/2024-jpp-facet-ii-correlations/
  - /publications/2024-nature-communications-lwir-plasma-acceleration/
  - /publications/2024-plasma-physics-control-fusion/
  - /publications/2024-pop-weibel-instability-magnetic-fields/
  - /publications/2024-prab-facet-ii-wakefield/
  - /publications/2024-prl-bidirectional-cascaded-lasing/
  - /publications/2025-nc-plasma-wakefield-transformer/
  - /publications/2026-PPCF-microbunching-LWFA/
---

<!-- _pages/publications.md -->

<p class="pub-total">{% bibliography_count %} peer-reviewed publications</p>

<style>
  /* room for the static [n] marker rendered by the bib template */
  .publications ol.bibliography > li {
    position: relative;
    padding-left: 3.4rem;
    margin-bottom: 2rem;
  }
  /* html prefix matches the specificity of the site-wide hide rule so this
     page's show rule wins by cascade order */
  html .publications .pub-num {
    display: block;
    position: absolute;
    left: 0;
    top: 0;
    font-weight: 600;
    color: var(--global-theme-color);
  }
</style>

<!-- Bibsearch Feature -->

{% include bib_search.liquid %}

<div class="publications">

{% bibliography %}

</div>

<script>
  /* Number entries [N]..[1] from document order, once. bibsearch only
     hides entries, so numbers stay correct under any filter. */
  (function () {
    var items = document.querySelectorAll(".publications ol.bibliography > li .pub-num");
    items.forEach(function (el, i) {
      el.textContent = "[" + (items.length - i) + "]";
    });
  })();
</script>
