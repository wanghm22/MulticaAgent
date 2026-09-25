The performance deficiency lies in computing capability, that is, insufficient computing power. The possible measures are as follows: 
Solution 1: Use vector instructions RVV to increase the number of elements computed per cycle; 
Solution 2: Use FMA to perform multiplication and addition together; 
Solution 3: Reduce precision in exchange for computing speed; 
Solution 4: If the computational complexity of the current algorithm is too high, consider switching to an algorithm with lower computational complexity; 
Solution 5: Avoid excessively long dependency chains through unrolling and rearrangement to improve ILP.
