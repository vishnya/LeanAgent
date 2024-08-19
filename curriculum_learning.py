import json
import math
import numpy as np
from collections import defaultdict

# We have easy, medium, hard, and hard (no proof)
# However, no proof should indicate that there is a "sorry" in the traced_tactics rather than traced_tactics being empty.
# Since theorems with empty traced_tactics (potentially using a term style proof) effectively don't matter for training or validation but do matter for prediction, we can
# just distribute them evenly across the easy, medium, and hard categories, but not the hard (no proof) category.

def calculate_difficulty(proof_steps):
    if any('sorry' in step.get('tactic') for step in proof_steps):
        return float('inf')  # Hard (no proof)
    if len(proof_steps) == 0:
        return None  # To be distributed later
    return math.exp(len(proof_steps))

def categorize_difficulty(difficulty, percentiles):
    if difficulty is None:
        return "To_Distribute"
    if difficulty == float('inf'):
        return "Hard (No proof)"
    elif difficulty <= percentiles[0]:
        return "Easy"
    elif difficulty <= percentiles[1]:
        return "Medium"
    else:
        return "Hard"

# file_path = "/raid/adarsh/datasets_new/new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5/corpus.jsonl"
# for line in open(file_path):
#     file_data = json.loads(line)

# {'path': '.lake/packages/lean4/src/lean/Init/Prelude.lean', 'imports': [], 'premises': [{'full_name': 'id', 'code': '@[inline] def id {α : Sort u} (a : α) : α := a', 'start': [21, 1], 'end': [33, 47], 'kind': 'commanddeclaration'}, {'full_name': 'Function.comp', 'code': '@[inline] def Function.comp {α : Sort u} {β : Sort v} {δ : Sort w} (f : β → δ) (g : α → β) : α → δ :=\n  fun x => f (g x)', 'start': [35, 1], 'end': [53, 19], 'kind': 'commanddeclaration'}, {'full_name': 'Function.const', 'code': '@[inline] def Function.const {α : Sort u} (β : Sort v) (a : α) : β → α :=\n  fun _ => a', 'start': [55, 1], 'end': [67, 13], 'kind': 'commanddeclaration'}, {'full_name': 'letFun', 'code': '@[irreducible] def letFun {α : Sort u} {β : α → Sort v} (v : α) (f : (x : α) → β x) : β v := f v', 'start': [69, 1], 'end': [80, 97], 'kind': 'commanddeclaration'}, {'full_name': 'inferInstance', 'code': 'abbrev inferInstance {α : Sort u} [i : α] : α := i', 'start': [83, 1], 'end': [99, 51], 'kind': 'commanddeclaration'}, {'full_name': 'inferInstanceAs', 'code': 'abbrev inferInstanceAs (α : Sort u) [i : α] : α := i', 'start': [102, 1], 'end': [113, 53], 'kind': 'commanddeclaration'}, {'full_name': 'PUnit', 'code': 'inductive PUnit : Sort u where\n  \n  | unit : PUnit', 'start': [116, 1], 'end': [124, 17], 'kind': 'commanddeclaration'}, {'full_name': 'Unit', 'code': 'abbrev Unit : Type := PUnit', 'start': [126, 1], 'end': [142, 28], 'kind': 'commanddeclaration'}, {'full_name': 'Unit.unit', 'code': '@[match_pattern] abbrev Unit.unit : Unit := PUnit.unit', 'start': [144, 1], 'end': [148, 55], 'kind': 'commanddeclaration'}, {'full_name': 'lcErased', 'code': 'unsafe axiom lcErased : Type', 'start': [150, 1], 'end': [151, 29], 'kind': 'commanddeclaration'}, {'full_name': 'lcProof', 'code': 'unsafe axiom lcProof {α : Prop} : α', 'start': [153, 1], 'end': [162, 36], 'kind': 'commanddeclaration'}, {'full_name': 'lcCast', 'code': 'unsafe axiom lcCast {α : Sort u} {β : Sort v} (a : α) : β', 'start': [164, 1], 'end': [167, 58], 'kind': 'commanddeclaration'}, {'full_name': 'lcUnreachable', 'code': 'unsafe axiom lcUnreachable {α : Sort u} : α', 'start': [170, 1], 'end': [182, 44], 'kind': 'commanddeclaration'}, {'full_name': 'True', 'code': 'inductive True : Prop where\n  \n  | intro : True', 'start': [184, 1], 'end': [192, 17], 'kind': 'commanddeclaration'}, {'full_name': 'False', 'code': 'inductive False : Prop', 'start': [194, 1], 'end': [202, 23], 'kind': 'commanddeclaration'}, {'full_name': 'Empty', 'code': 'inductive Empty : Type', 'start': [204, 1], 'end': [208, 23], 'kind': 'commanddeclaration'}, {'full_name': 'PEmpty', 'code': 'inductive PEmpty : Sort u where', 'start': [211, 1], 'end': [215, 32], 'kind': 'commanddeclaration'}, {'full_name': 'Not', 'code': 'def Not (a : Prop) : Prop := a → False', 'start': [217, 1], 'end': [224, 39], 'kind': 'commanddeclaration'}, {'full_name': 'False.elim', 'code': '@[macro_inline] def False.elim {C : Sort u} (h : False) : C :=\n  h.rec', 'start': [226, 1], 'end': [237, 8], 'kind': 'commanddeclaration'}, {'full_name': 'absurd', 'code': '@[macro_inline] def absurd {a : Prop} {b : Sort v} (h₁ : a) (h₂ : Not a) : b :=\n  (h₂ h₁).rec', 'start': [239, 1], 'end': [247, 14], 'kind': 'commanddeclaration'}, {'full_name': 'Eq', 'code': 'inductive Eq : α → α → Prop where\n  \n  | refl (a : α) : Eq a a', 'start': [249, 1], 'end': [279, 26], 'kind': 'commanddeclaration'}, {'full_name': 'Eq.ndrec', 'code': '@[simp] abbrev Eq.ndrec.{u1, u2} {α : Sort u2} {a : α} {motive : α → Sort u1} (m : motive a) {b : α} (h : Eq a b) : motive b :=\n  h.rec m', 'start': [281, 1], 'end': [283, 10], 'kind': 'commanddeclaration'}, {'full_name': 'rfl', 'code': '@[match_pattern] def rfl {α : Sort u} {a : α} : Eq a a := Eq.refl a', 'start': [285, 1], 'end': [294, 68], 'kind': 'commanddeclaration'}, {'full_name': 'id_eq', 'code': '@[simp] theorem id_eq (a : α) : Eq (id a) a', 'start': [296, 1], 'end': [297, 51], 'kind': 'commanddeclaration'}, {'full_name': 'Eq.subst', 'code': 'theorem Eq.subst {α : Sort u} {motive : α → Prop} {a b : α} (h₁ : Eq a b) (h₂ : motive a) : motive b', 'start': [299, 1], 'end': [313, 17], 'kind': 'commanddeclaration'}, {'full_name': 'Eq.symm', 'code': 'theorem Eq.symm {α : Sort u} {a b : α} (h : Eq a b) : Eq b a', 'start': [315, 1], 'end': [324, 10], 'kind': 'commanddeclaration'}, {'full_name': 'Eq.trans', 'code': 'theorem Eq.trans {α : Sort u} {a b c : α} (h₁ : Eq a b) (h₂ : Eq b c) : Eq a c', 'start': [326, 1], 'end': [336, 10], 'kind': 'commanddeclaration'}, {'full_name': 'cast', 'code': '@[macro_inline] def cast {α β : Sort u} (h : Eq α β) (a : α) : β :=\n  h.rec a', 'start': [338, 1], 'end': [350, 10], 'kind': 'commanddeclaration'}, {'full_name': 'congrArg', 'code': 'theorem congrArg {α : Sort u} {β : Sort v} {a₁ a₂ : α} (f : α → β) (h : Eq a₁ a₂) : Eq (f a₁) (f a₂)', 'start': [352, 1], 'end': [363, 10], 'kind': 'commanddeclaration'}, {'full_name': 'congr', 'code': 'theorem congr {α : Sort u} {β : Sort v} {f₁ f₂ : α → β} {a₁ a₂ : α} (h₁ : Eq f₁ f₂) (h₂ : Eq a₁ a₂) : Eq (f₁ a₁) (f₂ a₂)', 'start': [365, 1], 'end': [373, 16], 'kind': 'commanddeclaration'}, {'full_name': 'congrFun', 'code': 'theorem congrFun {α : Sort u} {β : α → Sort v} {f g : (x : α) → β x} (h : Eq f g) (a : α) : Eq (f a) (g a)', 'start': [375, 1], 'end': [377, 10], 'kind': 'commanddeclaration'}, {'full_name': 'Quot.lcInv', 'code': 'unsafe axiom Quot.lcInv {α : Sort u} {r : α → α → Prop} (q : Quot r) : α', 'start': [439, 1], 'end': [442, 73], 'kind': 'commanddeclaration'}, {'full_name': 'HEq', 'code': 'inductive HEq : {α : Sort u} → α → {β : Sort u} → β → Prop where\n  \n  | refl (a : α) : HEq a a', 'start': [444, 1], 'end': [458, 27], 'kind': 'commanddeclaration'}, {'full_name': 'HEq.rfl', 'code': '@[match_pattern] protected def HEq.rfl {α : Sort u} {a : α} : HEq a a :=\n  HEq.refl a', 'start': [460, 1], 'end': [462, 13], 'kind': 'commanddeclaration'}, {'full_name': 'eq_of_heq', 'code': "theorem eq_of_heq {α : Sort u} {a a' : α} (h : HEq a a') : Eq a a'", 'start': [464, 1], 'end': [468, 22], 'kind': 'commanddeclaration'}, {'full_name': 'Prod', 'code': 'structure Prod (α : Type u) (β : Type v) where\n  \n  mk ::\n  \n  fst : α\n  \n  snd : β', 'start': [470, 1], 'end': [485, 10], 'kind': 'commanddeclaration'}, {'full_name': 'PProd', 'code': '@[pp_using_anonymous_constructor]\nstructure PProd (α : Sort u) (β : Sort v) where\n  \n  fst : α\n  \n  snd : β', 'start': [489, 1], 'end': [498, 10], 'kind': 'commanddeclaration'}, {'full_name': 'MProd', 'code': 'structure MProd (α β : Type u) where\n  \n  fst : α\n  \n  snd : β', 'start': [500, 1], 'end': [508, 10], 'kind': 'commanddeclaration'}, {'full_name': 'And', 'code': '@[pp_using_anonymous_constructor]\nstructure And (a b : Prop) : Prop where\n  \n  intro ::\n  \n  left : a\n  \n  right : b', 'start': [510, 1], 'end': [524, 12], 'kind': 'commanddeclaration'}, {'full_name': 'Or', 'code': 'inductive Or (a b : Prop) : Prop where\n  \n  | inl (h : a) : Or a b\n  \n  | inr (h : b) : Or a b', 'start': [526, 1], 'end': [536, 25], 'kind': 'commanddeclaration'}, {'full_name': 'Or.intro_left', 'code': 'theorem Or.intro_left (b : Prop) (h : a) : Or a b', 'start': [538, 1], 'end': [540, 11], 'kind': 'commanddeclaration'}, {'full_name': 'Or.intro_right', 'code': 'theorem Or.intro_right (a : Prop) (h : b) : Or a b', 'start': [542, 1], 'end': [544, 11], 'kind': 'commanddeclaration'}, {'full_name': 'Or.elim', 'code': 'theorem Or.elim {c : Prop} (h : Or a b) (left : a → c) (right : b → c) : c', 'start': [546, 1], 'end': [553, 24], 'kind': 'commanddeclaration'}, {'full_name': 'Or.resolve_left', 'code': 'theorem Or.resolve_left  (h: Or a b) (na : Not a) : b', 'start': [555, 1], 'end': [555, 81], 'kind': 'commanddeclaration'}, {'full_name': 'Or.resolve_right', 'code': 'theorem Or.resolve_right (h: Or a b) (nb : Not b) : a', 'start': [556, 1], 'end': [556, 81], 'kind': 'commanddeclaration'}, {'full_name': 'Or.neg_resolve_left', 'code': 'theorem Or.neg_resolve_left  (h : Or (Not a) b) (ha : a) : b', 'start': [557, 1], 'end': [557, 86], 'kind': 'commanddeclaration'}, {'full_name': 'Or.neg_resolve_right', 'code': 'theorem Or.neg_resolve_right (h : Or a (Not b)) (nb : b) : a', 'start': [558, 1], 'end': [558, 86], 'kind': 'commanddeclaration'}, {'full_name': 'Bool', 'code': 'inductive Bool : Type where\n  \n  | false : Bool\n  \n  | true : Bool', 'start': [560, 1], 'end': [571, 16], 'kind': 'commanddeclaration'}, {'full_name': 'Subtype', 'code': '@[pp_using_anonymous_constructor]\nstructure Subtype {α : Sort u} (p : α → Prop) where\n  \n  val : α\n  \n  property : p val', 'start': [575, 1], 'end': [590, 19], 'kind': 'commanddeclaration'}, {'full_name': 'optParam', 'code': '@[reducible] def optParam (α : Sort u) (default : α) : Sort u := α', 'start': [593, 1], 'end': [600, 67], 'kind': 'commanddeclaration'}, {'full_name': 'outParam', 'code': '@[reducible] def outParam (α : Sort u) : Sort u := α', 'start': [602, 1], 'end': [618, 53], 'kind': 'commanddeclaration'}, {'full_name': 'semiOutParam', 'code': '@[reducible] def semiOutParam (α : Sort u) : Sort u := α', 'start': [620, 1], 'end': [641, 57], 'kind': 'commanddeclaration'}, {'full_name': 'namedPattern', 'code': '@[reducible] def namedPattern {α : Sort u} (x a : α) (h : Eq x a) : α := a', 'start': [644, 1], 'end': [645, 75], 'kind': 'commanddeclaration'}, {'full_name': 'sorryAx', 'code': '@[extern "lean_sorry", never_extract]\naxiom sorryAx (α : Sort u) (synthetic := false) : α', 'start': [647, 1], 'end': [664, 52], 'kind': 'commanddeclaration'}, {'full_name': 'eq_false_of_ne_true', 'code': 'theorem eq_false_of_ne_true : {b : Bool} → Not (Eq b true) → Eq b false', 'start': [666, 1], 'end': [668, 20], 'kind': 'commanddeclaration'}, {'full_name': 'eq_true_of_ne_false', 'code': 'theorem eq_true_of_ne_false : {b : Bool} → Not (Eq b false) → Eq b true', 'start': [670, 1], 'end': [672, 35], 'kind': 'commanddeclaration'}, ...

file_path = "/raid/adarsh/datasets_PT_full_merge_each_time/mathlib4_2b29e73438e240a427bcecc7c0fe19306beb1310/corpus.jsonl"
for line in open(file_path):
    file_data = json.loads(line)

# {"path": ".lake/packages/lean4/src/lean/Init/Prelude.lean", "imports": [], "premises": [{"full_name": "id", "code": "@[inline] def id {\\u03b1 : Sort u} (a : \\u03b1) : \\u03b1 := a", "start": [21, 1], "end": [33, 47], "kind": "commanddeclaration"}, {"full_name": "Function.comp", "code": "@[inline] def Function.comp {\\u03b1 : Sort u} {\\u03b2 : Sort v} {\\u03b4 : Sort w} (f : \\u03b2 \\u2192 \\u03b4) (g : \\u03b1 \\u2192 \\u03b2) : \\u03b1 \\u2192 \\u03b4 :=\\n  fun x => f (g x)", "start": [35, 1], "end": [53, 19], "kind": "commanddeclaration"}, {"full_name": "Function.const", "code": "@[inline] def Function.const {\\u03b1 : Sort u} (\\u03b2 : Sort v) (a : \\u03b1) : \\u03b2 \\u2192 \\u03b1 :=\\n  fun _ => a", "start": [55, 1], "end": [67, 13], "kind": "commanddeclaration"}, {"full_name": "letFun", "code": "@[irreducible] def letFun {\\u03b1 : Sort u} {\\u03b2 : \\u03b1 \\u2192 Sort v} (v : \\u03b1) (f : (x : \\u03b1) \\u2192 \\u03b2 x) : \\u03b2 v := f v", "start": [69, 1], "end": [80, 97], "kind": "commanddeclaration"}, {"full_name": "inferInstance", "code": "abbrev inferInstance {\\u03b1 : Sort u} [i : \\u03b1] : \\u03b1 := i", "start": [83, 1], "end": [99, 51], "kind": "commanddeclaration"}, {"full_name": "inferInstanceAs", "code": "abbrev inferInstanceAs (\\u03b1 : Sort u) [i : \\u03b1] : \\u03b1 := i", "start": [102, 1], "end": [113, 53], "kind": "commanddeclaration"} ...]

# file_path = "/raid/adarsh/datasets_new/new-version-test_f465306be03ced999caa157a85558a6c41b3e3f5/random/train.json"
# with open(file_path, 'r') as file:
#     data = json.load(file)

# 300
# {'url': 'https://github.com/Adarsh321123/new-version-test', 'commit': 'f465306be03ced999caa157a85558a6c41b3e3f5', 'file_path': '.lake/packages/mathlib/Mathlib/SetTheory/Ordinal/Arithmetic.lean', 'full_name': 'Ordinal.le_mul_right', 'theorem_statement': 'theorem le_mul_right (a : Ordinal) {b : Ordinal} (hb : 0 < b) : a ≤ b * a :=', 'start': [742, 1], 'end': [744, 17], 'traced_tactics': [{'tactic': "convert mul_le_mul_right' (one_le_iff_pos.2 hb) a", 'annotated_tactic': ["convert <a>mul_le_mul_right'</a> (<a>one_le_iff_pos</a>.2 hb) a", [{'full_name': "mul_le_mul_right'", 'def_path': '.lake/packages/mathlib/Mathlib/Algebra/Order/Monoid/Unbundled/Basic.lean', 'def_pos': [67, 9], 'def_end_pos': [67, 26]}, {'full_name': 'Ordinal.one_le_iff_pos', 'def_path': '.lake/packages/mathlib/Mathlib/SetTheory/Ordinal/Basic.lean', 'def_pos': [1063, 9], 'def_end_pos': [1063, 23]}]], 'state_before': 'α : Type u_1\nβ : Type u_2\nγ : Type u_3\nr : α → α → Prop\ns : β → β → Prop\nt : γ → γ → Prop\na b : Ordinal.{u_4}\nhb : 0 < b\n⊢ a ≤ b * a', 'state_after': "case h.e'_3\nα : Type u_1\nβ : Type u_2\nγ : Type u_3\nr : α → α → Prop\ns : β → β → Prop\nt : γ → γ → Prop\na b : Ordinal.{u_4}\nhb : 0 < b\n⊢ a = 1 * a"}, {'tactic': 'rw [one_mul a]', 'annotated_tactic': ['rw [<a>one_mul</a> a]', [{'full_name': 'one_mul', 'def_path': '.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [474, 9], 'def_end_pos': [474, 16]}]], 'state_before': "case h.e'_3\nα : Type u_1\nβ : Type u_2\nγ : Type u_3\nr : α → α → Prop\ns : β → β → Prop\nt : γ → γ → Prop\na b : Ordinal.{u_4}\nhb : 0 < b\n⊢ a = 1 * a", 'state_after': 'no goals'}]}

file_path = "/raid/adarsh/datasets_PT_full_merge_each_time/mathlib4_2b29e73438e240a427bcecc7c0fe19306beb1310/random/train.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# print(json.dumps(data[100], indent=2))
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Algebra/Ring/Subring/Basic.lean', 'full_name': 'Subring.closure_sUnion', 'start': [1024, 1], 'end': [1025, 27], 'traced_tactics': []}
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Order/BooleanAlgebra.lean', 'full_name': 'Bool.inf_eq_band', 'start': [831, 1], 'end': [832, 6], 'traced_tactics': []}
# 100 (1)
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Algebra/Group/Basic.lean', 'full_name': 'one_div_mul_one_div', 'start': [752, 1], 'end': [752, 71], 'traced_tactics': [{'tactic': 'simp', 'annotated_tactic': ['simp', []], 'state_before': 'α : Type u_1\nβ : Type u_2\nG : Type u_3\nM : Type u_4\ninst✝ : DivisionCommMonoid α\na b c d : α\n⊢ 1 / a * (1 / b) = 1 / (a * b)', 'state_after': 'no goals'}]}
# 300 (8)
# {'url': 'https://github.com/teorth/pfr', 'commit': '6a5082ee465f9e44cea479c7b741b3163162bb7e', 'file_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'full_name': 'ContinuousLinearMap.opNorm_lsmul', 'start': [281, 1], 'end': [290, 26], 'traced_tactics': [{'tactic': "refine' ContinuousLinearMap.opNorm_eq_of_bounds zero_le_one (fun x => _) fun N _ h => _", 'annotated_tactic': ["refine' <a>ContinuousLinearMap.opNorm_eq_of_bounds</a> <a>zero_le_one</a> (fun x => _) fun N _ h => _", [{'full_name': 'ContinuousLinearMap.opNorm_eq_of_bounds', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Basic.lean', 'def_pos': [180, 9], 'def_end_pos': [180, 28]}, {'full_name': 'zero_le_one', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Order/ZeroLEOne.lean', 'def_pos': [26, 15], 'def_end_pos': [26, 26]}]], 'state_before': "𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\n⊢ ‖lsmul 𝕜 𝕜'‖ = 1", 'state_after': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ 1 * ‖x‖\n\ncase refine'_2\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\n⊢ 1 ≤ N"}, {'tactic': 'obtain ⟨y, hy⟩ := exists_ne (0 : E)', 'annotated_tactic': ['obtain ⟨y, hy⟩ := <a>exists_ne</a> (0 : E)', [{'full_name': 'exists_ne', 'def_path': './.lake/packages/mathlib/Mathlib/Logic/Nontrivial/Defs.lean', 'def_pos': [53, 9], 'def_end_pos': [53, 18]}]], 'state_before': "case refine'_2\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\n⊢ 1 ≤ N"}, {'tactic': 'have := le_of_opNorm_le _ (h 1) y', 'annotated_tactic': ['have := <a>le_of_opNorm_le</a> _ (h 1) y', [{'full_name': 'ContinuousLinearMap.le_of_opNorm_le', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Basic.lean', 'def_pos': [243, 9], 'def_end_pos': [243, 24]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖((lsmul 𝕜 𝕜') 1) y‖ ≤ N * ‖1‖ * ‖y‖\n⊢ 1 ≤ N"}, {'tactic': 'simp_rw [lsmul_apply, one_smul, norm_one, mul_one] at this', 'annotated_tactic': ['simp_rw [<a>lsmul_apply</a>, <a>one_smul</a>, <a>norm_one</a>, <a>mul_one</a>] at this', [{'full_name': 'ContinuousLinearMap.lsmul_apply', 'def_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'def_pos': [212, 9], 'def_end_pos': [212, 20]}, {'full_name': 'one_smul', 'def_path': './.lake/packages/mathlib/Mathlib/GroupTheory/GroupAction/Defs.lean', 'def_pos': [483, 9], 'def_end_pos': [483, 17]}, {'full_name': 'NormOneClass.norm_one', 'def_path': './.lake/packages/mathlib/Mathlib/Analysis/Normed/Field/Basic.lean', 'def_pos': [165, 3], 'def_end_pos': [165, 11]}, {'full_name': 'mul_one', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [480, 9], 'def_end_pos': [480, 16]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖((lsmul 𝕜 𝕜') 1) y‖ ≤ N * ‖1‖ * ‖y‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 ≤ N"}, {'tactic': "refine' le_of_mul_le_mul_right _ (norm_pos_iff.mpr hy)", 'annotated_tactic': ["refine' <a>le_of_mul_le_mul_right</a> _ (norm_pos_iff.mpr hy)", [{'full_name': 'le_of_mul_le_mul_right', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Order/Ring/Lemmas.lean', 'def_pos': [236, 9], 'def_end_pos': [236, 31]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 ≤ N", 'state_after': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 * ‖y‖ ≤ N * ‖y‖"}, {'tactic': 'simp_rw [one_mul, this]', 'annotated_tactic': ['simp_rw [<a>one_mul</a>, this]', [{'full_name': 'one_mul', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [474, 9], 'def_end_pos': [474, 16]}]], 'state_before': "case refine'_2.intro\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nN : ℝ\nx✝ : N ≥ 0\nh : ∀ (x : 𝕜'), ‖(lsmul 𝕜 𝕜') x‖ ≤ N * ‖x‖\ny : E\nhy : y ≠ 0\nthis : ‖y‖ ≤ N * ‖y‖\n⊢ 1 * ‖y‖ ≤ N * ‖y‖", 'state_after': 'no goals'}, {'tactic': 'rw [one_mul]', 'annotated_tactic': ['rw [<a>one_mul</a>]', [{'full_name': 'one_mul', 'def_path': './.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean', 'def_pos': [474, 9], 'def_end_pos': [474, 16]}]], 'state_before': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ 1 * ‖x‖", 'state_after': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ ‖x‖"}, {'tactic': 'apply opNorm_lsmul_apply_le', 'annotated_tactic': ['apply <a>opNorm_lsmul_apply_le</a>', [{'full_name': 'ContinuousLinearMap.opNorm_lsmul_apply_le', 'def_path': '.lake/packages/mathlib/Mathlib/Analysis/NormedSpace/OperatorNorm/Mul.lean', 'def_pos': [228, 9], 'def_end_pos': [228, 30]}]], 'state_before': "case refine'_1\n𝕜 : Type u_1\nE : Type u_2\ninst✝⁷ : NontriviallyNormedField 𝕜\ninst✝⁶ : NormedAddCommGroup E\ninst✝⁵ : NormedSpace 𝕜 E\nc : 𝕜\n𝕜' : Type u_3\ninst✝⁴ : NormedField 𝕜'\ninst✝³ : NormedAlgebra 𝕜 𝕜'\ninst✝² : NormedSpace 𝕜' E\ninst✝¹ : IsScalarTower 𝕜 𝕜' E\ninst✝ : Nontrivial E\nx : 𝕜'\n⊢ ‖(lsmul 𝕜 𝕜') x‖ ≤ ‖x‖", 'state_after': 'no goals'}]}

difficulties_by_url = defaultdict(list)

for item in data:
    proof_steps = item['traced_tactics']
    difficulty = calculate_difficulty(proof_steps)
    difficulties_by_url[item['url']].append((item['full_name'], item['file_path'], difficulty))

all_difficulties = [diff for url_difficulties in difficulties_by_url.values() 
                    for _, _, diff in url_difficulties if diff is not None]

percentiles = np.percentile(all_difficulties, [33, 67])

categorized_theorems = defaultdict(lambda: defaultdict(list))

for url, theorems in difficulties_by_url.items():
    for theorem_name, file_path, difficulty in theorems:
        category = categorize_difficulty(difficulty, percentiles)
        categorized_theorems[url][category].append((theorem_name, file_path, difficulty))

for url in categorized_theorems:
    to_distribute = categorized_theorems[url]["To_Distribute"]
    chunk_size = len(to_distribute) // 3
    for i, category in enumerate(["Easy", "Medium", "Hard"]):
        start = i * chunk_size
        end = start + chunk_size if i < 2 else None
        categorized_theorems[url][category].extend(to_distribute[start:end])
    del categorized_theorems[url]["To_Distribute"]

print("Summary of theorem difficulties by URL:")
for url, categories in categorized_theorems.items():
    print(f"\nURL: {url}")
    for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
        theorems = categories[category]
        print(f"  {category}: {len(theorems)} theorems")
        if theorems:
            sorted_theorems = sorted(theorems, key=lambda x: x[2] if x[2] is not None else -float('inf'), reverse=True)[:3]
            for name, path, diff in sorted_theorems:
                diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                print(f"    - {name} (File: {path}, Difficulty: {diff_str})")

print("\nOverall Statistics:")
total_theorems = sum(len(theorems) for categories in categorized_theorems.values() for theorems in categories.values())
for category in ["Easy", "Medium", "Hard", "Hard (No proof)"]:
    count = sum(len(categories[category]) for categories in categorized_theorems.values())
    percentage = (count / total_theorems) * 100
    print(f"{category}: {count} theorems ({percentage:.2f}%)")

print(f"\nPercentile thresholds: Easy <= {percentiles[0]:.2f}, Medium <= {percentiles[1]:.2f}, Hard > {percentiles[1]:.2f}")

# TODO: filter out dependencies by comparing to name of each repo we merged
print("\nTheorems without proofs (Hard (No proof)), sorted by file path:")
for url, categories in categorized_theorems.items():
    no_proof_theorems = categories["Hard (No proof)"]
    if no_proof_theorems:
        print(f"\nURL: {url}")
        print(f"  Hard (No proof): {len(no_proof_theorems)} theorems")
        sorted_theorems = sorted(no_proof_theorems, key=lambda x: x[1])  # Sort by file path
        for name, path, diff in sorted_theorems:
            print(f"  - {name} (File: {path})")

# Summary of theorem difficulties by URL:

# URL: https://github.com/teorth/pfr
#   Easy: 25827 theorems
#     - pow_eq_one_iff_cases (File: .lake/packages/mathlib/Mathlib/Data/Nat/Parity.lean, Difficulty: 2.72)
#     - List.countP_cons_of_neg (File: .lake/packages/batteries/Batteries/Data/List/Count.lean, Difficulty: 2.72)
#     - Real.cauchy_add (File: .lake/packages/mathlib/Mathlib/Data/Real/Basic.lean, Difficulty: 2.72)
#   Medium: 23512 theorems
#     - List.IsRotated.map (File: .lake/packages/mathlib/Mathlib/Data/List/Rotate.lean, Difficulty: 20.09)
#     - Finset.sup'_image (File: .lake/packages/mathlib/Mathlib/Data/Finset/Lattice.lean, Difficulty: 20.09)
#     - Submodule.snd_map_snd (File: .lake/packages/mathlib/Mathlib/LinearAlgebra/Prod.lean, Difficulty: 20.09)
#   Hard: 21951 theorems
#     - ProbabilityTheory.measureMutualInfo_nonneg_aux (File: PFR/ForMathlib/Entropy/Measure.lean, Difficulty: 47445721460229658823571144923127459653286734449467784541564019054311377797245669985431549116416.00)
#     - rdist_of_neg_le (File: PFR/MoreRuzsaDist.lean, Difficulty: 183804612428282463707433348561087313620821365041237266589308369031523205120.00)
#     - weak_PFR_asymm_prelim (File: PFR/WeakPFR.lean, Difficulty: 8344716494264774957669830503803807239792132249251932789959986398625792.00)
#   Hard (No proof): 42 theorems
#     - ProbabilityTheory.kernel.iIndepFun.finsets (File: PFR/Mathlib/Probability/Independence/Kernel.lean, Difficulty: inf)
#     - rho_of_sum_le (File: PFR/RhoFunctional.lean, Difficulty: inf)
#     - multidist_ruzsa_I (File: PFR/MoreRuzsaDist.lean, Difficulty: inf)

# Overall Statistics:
# Easy: 25827 theorems (36.21%)
# Medium: 23512 theorems (32.96%)
# Hard: 21951 theorems (30.77%)
# Hard (No proof): 42 theorems (0.06%)

# Percentile thresholds: Easy <= 2.72, Medium <= 20.09, Hard > 20.09

# Theorems without proofs (Hard (No proof)), sorted by file path:

# URL: https://github.com/teorth/pfr
#   Hard (No proof): 42 theorems
#   - ProbabilityTheory.kernel.iIndepFun.finsets (File: PFR/Mathlib/Probability/Independence/Kernel.lean)
#   - multidist_ruzsa_I (File: PFR/MoreRuzsaDist.lean)
#   - multiDist_of_perm (File: PFR/MoreRuzsaDist.lean)
#   - multidist_eq_zero (File: PFR/MoreRuzsaDist.lean)
#   - iter_multiDist_chainRule' (File: PFR/MoreRuzsaDist.lean)
#   - cond_multiDist_chainRule (File: PFR/MoreRuzsaDist.lean)
#   - multiDist_chainRule (File: PFR/MoreRuzsaDist.lean)
#   - multidist_ruzsa_II (File: PFR/MoreRuzsaDist.lean)
#   - iter_multiDist_chainRule (File: PFR/MoreRuzsaDist.lean)
#   - multiDist_nonneg (File: PFR/MoreRuzsaDist.lean)
#   - multidist_ruzsa_IV (File: PFR/MoreRuzsaDist.lean)
#   - ent_of_sum_le_ent_of_sum (File: PFR/MoreRuzsaDist.lean)
#   - cor_multiDist_chainRule (File: PFR/MoreRuzsaDist.lean)
#   - multiDist_indep (File: PFR/MoreRuzsaDist.lean)
#   - multidist_ruzsa_III (File: PFR/MoreRuzsaDist.lean)
#   - multiTau_continuous (File: PFR/MultiTauFunctional.lean)
#   - sub_condMultiDistance_le' (File: PFR/MultiTauFunctional.lean)
#   - sub_condMultiDistance_le (File: PFR/MultiTauFunctional.lean)
#   - multiTau_min_exists (File: PFR/MultiTauFunctional.lean)
#   - sub_multiDistance_le (File: PFR/MultiTauFunctional.lean)
#   - multiTau_min_sum_le (File: PFR/MultiTauFunctional.lean)
#   - rho_of_sum_le (File: PFR/RhoFunctional.lean)
#   - condRho_le (File: PFR/RhoFunctional.lean)
#   - condRho_sum_le' (File: PFR/RhoFunctional.lean)
#   - rho_of_translate (File: PFR/RhoFunctional.lean)
#   - condRho_of_injective (File: PFR/RhoFunctional.lean)
#   - rho_of_subgroup (File: PFR/RhoFunctional.lean)
#   - I_one_le (File: PFR/RhoFunctional.lean)
#   - rho_plus_of_sum (File: PFR/RhoFunctional.lean)
#   - dist_of_min_eq_zero (File: PFR/RhoFunctional.lean)
#   - dist_le_of_sum_zero (File: PFR/RhoFunctional.lean)
#   - condRho_minus_le (File: PFR/RhoFunctional.lean)
#   - condRho_of_translate (File: PFR/RhoFunctional.lean)
#   - condRho_plus_le (File: PFR/RhoFunctional.lean)
#   - rho_minus_of_sum (File: PFR/RhoFunctional.lean)
#   - condRho_of_sum_le (File: PFR/RhoFunctional.lean)
#   - condRho_sum_le (File: PFR/RhoFunctional.lean)
#   - rho_continuous (File: PFR/RhoFunctional.lean)
#   - dist_add_dist_eq (File: PFR/RhoFunctional.lean)
#   - I_two_le (File: PFR/RhoFunctional.lean)
#   - rho_of_sum (File: PFR/RhoFunctional.lean)
#   - phi_min_exists (File: PFR/RhoFunctional.lean)