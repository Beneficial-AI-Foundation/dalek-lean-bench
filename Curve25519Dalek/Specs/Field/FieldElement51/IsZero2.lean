/-
Copyright (c) 2025 Beneficial AI Foundation. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Markus Dablander, Hoang Le Truong
-/
import Curve25519Dalek.Funs
import Curve25519Dalek.Math.Basic
import Curve25519Dalek.Aux
import Curve25519Dalek.Specs.Backend.Serial.U64.Field.FieldElement51.Reduce
import Curve25519Dalek.Specs.Backend.Serial.U64.Field.FieldElement51.ToBytes

set_option linter.style.longLine false
set_option linter.style.setOption false

/-!
# Spec Theorem for `FieldElement51::is_zero`

Specification and proof for `FieldElement51::is_zero`.

This function checks whether a field element is zero.

**Source**: curve25519-dalek/src/field.rs
-/

open Aeneas Aeneas.Std Result Aeneas.Std.WP

namespace curve25519_dalek.field.FieldElement51

/-!
Natural language description:

- Checks whether a field element is zero in 𝔽_p, p = 2^255 - 19.
- Field element is in radix 2^51 with five u64 limbs.
- Returns a `subtle.Choice` (0 = false, 1 = true).

Spec:

- Function succeeds (no panic).
- For any field element `r`, result `c` satisfies
  `c.val = 1 ↔ Field51_as_Nat(r) % p = 0`.
-/

/-- Arrays are equal if their slices are equal. -/
lemma array_eq_of_to_slice_eq {α : Type} {n : Usize} {h1 h2 : Array α n}
    (h : h1.to_slice = h2.to_slice) :
    h1 = h2 := by
  simp only [Array.to_slice] at h
  cases h1; cases h2
  simp only at h
  cases h
  rfl

/-! ## Workaround for `progress` timeout

The `progress` tactic runs `simp` on **all hypotheses** in the context. After `progress`
processes `to_bytes`, the postconditions involving `U8x32_as_Nat` / `Field51_as_Nat` are
expensive for `simp` to reduce (whnf timeout), causing any subsequent `progress` call to
time out.

Workaround:
1. Use `simp only [lift, bind_tc_ok] at ⊢` to reduce the pure `Array.to_slice` binds
   without touching hypotheses (bypasses `progress` for those steps).
2. Wrap the expensive hypotheses in `Hold` (opaque to `simp`) before calling `progress`
   on `ct_eq`.
3. Recover with `change ... at` afterwards (`Hold P` is defeq to `P`).
-/

private def Hold (P : Prop) : Prop := P

@[progress]
theorem is_zero_spec (r : backend.serial.u64.field.FieldElement51) :
    is_zero r ⦃ c =>
    (c.val = 1#u8 ↔ Field51_as_Nat r % p = 0) ⦄ := by
  sorry

end curve25519_dalek.field.FieldElement51
