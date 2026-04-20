# Attachment Sim

A biologically-modelled attachment theory simulation built in Blazor WebAssembly.

Agents have a four-region cortico-limbic brain (Amygdala, Hippocampus, Prefrontal Cortex,
Anterior Cingulate Cortex), each modelled as an independent leaky integrate-and-fire neural
network. Attachment styles emerge from caregiver responsiveness during the critical period
and propagate across infinite generations via intergenerational weight inheritance.

Based on: Bowlby (1969), Ainsworth (1978), Hebb (1949)

## Running locally
dotnet restore
dotnet run

## Deploy to GitHub Pages
Push to main branch. Enable GitHub Actions in Settings > Pages.
Live at: https://YOURNAME.github.io/YOURREPO/
