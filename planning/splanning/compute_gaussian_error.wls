#!/usr/bin/env wolframscript
(* ::Package:: *)

(*Make the error for the 3d gaussian*)
f[s_]:=N[1-Integrate[PDF[MultinormalDistribution[IdentityMatrix[3]],{x,y,z}], {x,y,z} \[Element] Ball[{0,0,0},s]],50]
(*Results degenerate past 8. save file*)
SetDirectory[NotebookDirectory[]]
Export["gaussian_table.csv", Table[{s,f[s]},{s,2,8.5,0.02}], "CSV"]
