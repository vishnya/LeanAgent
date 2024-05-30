open Nat (add_assoc add_comm)

theorem foo (a : Nat) : a + 1 = Nat.succ a := by
  rfl

theorem bar (a b : Nat) : a + b = b + a := by
  rw[add_comm]
