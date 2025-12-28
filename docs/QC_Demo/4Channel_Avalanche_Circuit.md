# 4-Channel Avalanche Entropy Circuit

## Design Philosophy

Following Chris Tacke's insight: **don't over-engineer it**. The ADC reads the raw noise, software handles the rest. Each channel is identical and isolated.

## Block Diagram

```
+5V Rail ──┬────────────┬────────────┬────────────┬────────────┐
           │            │            │            │            │
         ┌─┴─┐        ┌─┴─┐        ┌─┴─┐        ┌─┴─┐         │
         │470│        │470│        │470│        │470│       ┌─┴─┐
         │ Ω │        │ Ω │        │ Ω │        │ Ω │       │LM │
         └─┬─┘        └─┬─┘        └─┬─┘        └─┬─┘       │324│
           │            │            │            │         │   │
          ─┴─          ─┴─          ─┴─          ─┴─        │Vcc│
         ╲   ╱        ╲   ╱        ╲   ╱        ╲   ╱       └─┬─┘
          ╲ ╱ Z1       ╲ ╱ Z2       ╲ ╱ Z3       ╲ ╱ Z4       │
           │            │            │            │           │
           ├──→ BUF1    ├──→ BUF2    ├──→ BUF3    ├──→ BUF4   │
           │            │            │            │           │
          ─┴─          ─┴─          ─┴─          ─┴─          │
          GND          GND          GND          GND          │
                                                              │
                        ┌─────────────────────────────────────┘
                        │
                      ──┴──
                       GND
```

## Schematic - Single Channel (x4)

```
                    +5V
                     │
                   ┌─┴─┐
                   │R1 │ 470Ω (current limit)
                   │   │
                   └─┬─┘
                     │
                     ├────────────────┐
                     │                │
                    ─┴─             ┌─┴─┐
                   ╲   ╱            │C1 │ 100nF (HF bypass)
                    ╲ ╱  BZX55C3V3  │   │
                     │   (reverse)  └─┬─┘
                     │                │
                     │    ┌───────────┤
                     │    │           │
                     │  ┌─┴─┐        ─┴─
                     │  │R2 │ 10kΩ   GND
                     │  │   │ (pull-down)
                     │  └─┬─┘
                     │    │
                     └────┼──────────────────┐
                          │                  │
                         ─┴─               ┌─┴─┐
                         GND               │   │+
                                           │ A ├────────→ To ADC
                                     ┌────►│   │
                                     │     └─┬─┘
                                     │       │
                                     └───────┘
                                    (voltage follower)
```

## Component List

### Per Channel (x4)
| Ref | Value | Description |
|-----|-------|-------------|
| R1 | 470Ω | Zener current limiter (~3.4mA at 5V-3.3V) |
| R2 | 10kΩ | Pull-down (ensures defined DC level) |
| C1 | 100nF | HF noise bypass (ceramic) |
| Z1 | BZX55C3V3 | 3.3V Zener (avalanche source) |

### Shared
| Ref | Value | Description |
|-----|-------|-------------|
| U1 | LM324 | Quad op-amp (4 buffers in 1 package) |
| C2 | 100µF | Power supply bulk decoupling |
| C3 | 100nF | Power supply HF decoupling |

### Total BOM
| Qty | Part | Notes |
|-----|------|-------|
| 4 | BZX55C3V3 | Or any 3.3V Zener in similar package |
| 4 | 470Ω 1/4W | 5% tolerance fine |
| 4 | 10kΩ 1/4W | 5% tolerance fine |
| 5 | 100nF ceramic | 4 for channels + 1 for power |
| 1 | 100µF electrolytic | 10V+ rating |
| 1 | LM324N | DIP-14 quad op-amp |

## LM324 Pinout & Wiring

```
           ┌───────────┐
    OUT1 ──┤1       14├── +Vcc (+5V)
     -1  ──┤2       13├── OUT4
     +1  ──┤3       12├── -4
    GND  ──┤4       11├── +4
     +2  ──┤5       10├── -3
     -2  ──┤6        9├── OUT3
    OUT2 ──┤7        8├── +3
           └───────────┘

Wiring per amplifier (voltage follower):
  + input ← Zener junction
  - input ← OUT (feedback)
  OUT     → ADC channel
```

## Physical Layout Suggestion

```
┌─────────────────────────────────────────────────────────────────┐
│  +5V ●────────────────────────────────────────────────────●     │
│      │                                                    │     │
│    ┌─┴─┐  ┌─────────────────────────────────────────┐   ─┼─    │
│    │100│  │              LM324                      │  100µF   │
│    │ nF│  │  ┌───┐  ┌───┐  ┌───┐  ┌───┐            │    │     │
│    └─┬─┘  │  │ A1│  │ A2│  │ A3│  │ A4│            │   ─┴─    │
│      │    │  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘            │   GND    │
│      │    └────┼──────┼──────┼──────┼──────────────┘          │
│      │         │      │      │      │                         │
│  ────┴─────────┴──────┴──────┴──────┴─────────────────────    │
│                                                          GND   │
│                                                                │
│   CH0          CH1          CH2          CH3                   │
│  ┌────┐       ┌────┐       ┌────┐       ┌────┐                │
│  │470Ω│       │470Ω│       │470Ω│       │470Ω│                │
│  └──┬─┘       └──┬─┘       └──┬─┘       └──┬─┘                │
│     │            │            │            │                   │
│    ─┴─          ─┴─          ─┴─          ─┴─                  │
│   Z 3V3        Z 3V3        Z 3V3        Z 3V3                 │
│    ─┬─          ─┬─          ─┬─          ─┬─                  │
│     │┌──┐        │┌──┐        │┌──┐        │┌──┐               │
│     ├┤nF│        ├┤nF│        ├┤nF│        ├┤nF│               │
│     │└──┘        │└──┘        │└──┘        │└──┘               │
│     │┌──┐        │┌──┐        │┌──┐        │┌──┐               │
│     ├┤10k        ├┤10k        ├┤10k        ├┤10k               │
│     │└──┘        │└──┘        │└──┘        │└──┘               │
│     │            │            │            │                   │
│  ───┴────────────┴────────────┴────────────┴──────────────     │
│                                                          GND   │
│                                                                │
│   To LM324      To LM324      To LM324      To LM324           │
│   +1 (pin3)     +2 (pin5)     +3 (pin8)     +4 (pin11)         │
│                                                                │
│   OUT1→CH0      OUT2→CH1      OUT3→CH2      OUT4→CH3           │
│   (pin1)        (pin7)        (pin9)        (pin13)            │
│      ↓             ↓             ↓             ↓               │
│   MCP3004       MCP3004       MCP3004       MCP3004            │
│   Pin 2         Pin 3         Pin 4         Pin 5              │
└─────────────────────────────────────────────────────────────────┘
```

## Connection to YoshiPi

```
4-Channel Board          YoshiPi MCP3004
─────────────────        ───────────────
+5V  ────────────────────  5V (pin 2 or 4)
GND  ────────────────────  GND (pin 6, 9, 14, etc.)
CH0 OUT ─────────────────  CH0 (MCP3004 pin 2)
CH1 OUT ─────────────────  CH1 (MCP3004 pin 3)
CH2 OUT ─────────────────  CH2 (MCP3004 pin 4)
CH3 OUT ─────────────────  CH3 (MCP3004 pin 5)
```

## Why This Design

1. **Simple**: Same 4 components per channel, one shared quad op-amp
2. **Buffered**: Op-amp followers prevent ADC loading from affecting noise
3. **Decoupled**: Each Zener has its own current source, no shared paths
4. **Isolated**: 100nF caps filter shared supply noise
5. **Consistent**: Identical channels = identical entropy characteristics

## Expected Output

Each channel should produce:
- DC level: ~3.3V (Zener voltage)
- Noise: Avalanche noise riding on DC
- ADC reading: ~650-700 counts center, ±100-300 counts noise swing
- LSB quality: Balanced bits 0-2 at minimum

## Assembly Notes

1. Keep Zener leads short (minimize pickup)
2. Star ground if possible (all grounds to single point)
3. Place 100nF caps close to their respective Zeners
4. Place power caps close to LM324 Vcc pin
5. Consider shielding if in noisy environment

## Testing Sequence

1. Power up, measure Zener voltages (~3.3V each)
2. Measure op-amp outputs (~3.3V each, should match Zener)
3. Run `python3 ~/bin/validate_4ch.py` on YoshiPi
4. All 4 channels should show similar range and balance
