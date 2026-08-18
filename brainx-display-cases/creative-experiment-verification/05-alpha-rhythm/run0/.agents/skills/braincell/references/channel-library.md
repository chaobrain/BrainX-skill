# BrainCell channel library

Use this reference when selecting a built-in membrane channel, matching it to its ion owner, or designing a controlled channel comparison. Author a custom channel only after the built-in library and literature-derived templates have been checked.

## Mental model

A channel computes membrane current from voltage, its internal gating state, and the `IonInfo` supplied by its `root_type`.

Built-in names usually follow `<current>_<source><year>`, so a similar current name does not imply interchangeable kinetics. Match the requested model, paper, temperature convention, reversal-potential treatment, and ion dependency before tuning `g_max`.

## Choose a channel family

### Sodium channels

Use sodium channels for inward current that initiates or sustains depolarization.

| API | Description |
|---|---|
| `braincell.channel.Na_HH1952(size, ...)` | Use for the classical Hodgkin-Huxley 1952 sodium current. |
| `braincell.channel.Na_TM1991(size, ...)` | Use when reproducing the Traub-Miles 1991 sodium formulation. |
| `braincell.channel.Na_Ba2002(size, ...)` | Use for models based on the Bazhenov 2002 sodium current. |
| `braincell.channel.NaF_SU2015_DCN(size, ...)` | Use when the requested model requires the fast sodium current from the Sudhakar 2015 DCN template family. |
| `braincell.channel.NaP_SU2015_DCN(size, ...)` | Use when the requested model requires the persistent sodium current from the same template family. |

Attach these channels to a sodium ion. Do not select among them by maximal conductance alone; their gating equations and source-model conventions differ.

### Potassium channels

Use potassium channels for repolarization, resting conductance, delayed rectification, transient A-current, M-current, or inward rectification.

| API | Description |
|---|---|
| `braincell.channel.K_HH1952(size, ...)` | Use for the classical Hodgkin-Huxley 1952 delayed-rectifier potassium current. |
| `braincell.channel.K_TM1991(size, ...)` | Use when reproducing the Traub-Miles 1991 potassium formulation. |
| `braincell.channel.K_Leak(size, ...)` | Use for an ion-specific potassium leak current owned by a potassium ion. |
| `braincell.channel.KDR_Ba2002(size, ...)` | Use for the Bazhenov 2002 delayed-rectifier potassium current. |
| `braincell.channel.KA1_HM1992(size, ...)` | Use for the Huguenard-McCormick 1992 `IA1` transient potassium current. |
| `braincell.channel.KA2_HM1992(size, ...)` | Use for the Huguenard-McCormick 1992 `IA2` transient potassium current. |

The API also exposes `KM_*`, `Kv*`, and `Kir*` template families. Use the exact literature-derived class when the user names a model or source paper; consult the installed API for the complete list.

### Calcium channels

Use calcium channels for calcium entry, low- or high-threshold activation, rebound behavior, or calcium-dependent downstream mechanisms.

| API | Description |
|---|---|
| `braincell.channel.CaL_IS2008(size, ...)` | Use for the Inoue-Strowbridge 2008 L-type calcium current. |
| `braincell.channel.CaN_IS2008(size, ...)` | Use for the Inoue-Strowbridge 2008 calcium-activated non-selective cation current. |
| `braincell.channel.CaT_HM1992(size, ...)` | Use for the Huguenard-McCormick 1992 low-threshold T-type calcium current. |
| `braincell.channel.CaT_HP1992(size, ...)` | Use for the Huguenard-Prince 1992 T-type current for reticular-nucleus models. |
| `braincell.channel.CaHT_HM1992(size, ...)` | Use for the Huguenard-McCormick 1992 high-threshold calcium current. |

Attach calcium channels to a calcium ion. Use a dynamic calcium ion when accumulated calcium must affect reversal potential or calcium-dependent channels; otherwise a fixed calcium ion may be sufficient.

### Mixed-ion, HCN, and leak channels

| API | Description |
|---|---|
| `braincell.channel.AHP_De1994(size, ...)` | Use for a calcium-activated potassium AHP current; attach it to `MixIons(k, ca)` in that order. |
| `braincell.channel.HCN_HM1992(size, E=..., ...)` | Use for the Huguenard-McCormick 1992 hyperpolarization-activated current and related sag or rebound behavior. |
| `braincell.channel.IL(size, E=..., g_max=...)` | Use for a root-cell leakage current with its own reversal potential; attach it directly to the cell. |
| `braincell.channel.LeakageChannel` | Use as the base class when implementing leakage-channel dynamics, not as the normal concrete leak mechanism. |

The `Kca*`, `HCN*`, and template-import families contain model-specific alternatives. Prefer the exact source match over a name that merely has the same broad current type.


## Route custom authoring

Open `references/braincell-custom-ion-channel-authoring.md` only when:

- no built-in or literature-derived class implements the required kinetics;
- the requested ion dependency can be stated as a correct `root_type`;
- the governing equations, units, initialization rule, and validation target are known.

For a multicompartment task, establish the `Morphology -> Cell -> paint/place -> init_state` workflow in `references/multicompartment/multicompartment-cell-workflow.md` before reusing the custom-authoring route.


## Sources

- Ions & Channels: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html
- Channels tutorial: https://brainx.chaobrain.com/braincell/tutorials/channel.html
- `braincell.channel` API: https://brainx.chaobrain.com/braincell/apis/braincell.channel.html
- Channel ablation: https://brainx.chaobrain.com/braincell/examples/channel_ablation.html
- Spike-frequency adaptation: https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html
