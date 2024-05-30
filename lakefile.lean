import Lake
open Lake DSL

package «ReProver» where
  -- add package configuration options here

lean_lib «ReProver» where
  -- add library configuration options here

@[default_target]
lean_exe «reprover» where
  root := `Main
