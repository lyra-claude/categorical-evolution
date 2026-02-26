module Main (main) where

import System.Exit (exitFailure, exitSuccess)
import Test.Category
import Test.Operators
import Test.Pipeline

main :: IO ()
main = do
  putStrLn "=== categorical-evolution test suite ==="
  putStrLn ""
  r1 <- Test.Category.runTests
  r2 <- Test.Operators.runTests
  r3 <- Test.Pipeline.runTests
  putStrLn ""
  let total = r1 + r2 + r3
  if total == 0
    then putStrLn "All tests passed!" >> exitSuccess
    else putStrLn (show total ++ " test(s) failed.") >> exitFailure
