Project-2
=========

A Study in Parallel Algorithms : Stream Compaction

![](https://github.com/DiracSea3921/Project2-StreamCompaction/blob/master/test1.png)

When the data set is small, paraller computation takes more time. When array's length is more than 1000000, parallel coputation become more efficient
My shared memory version may have some problem, the time taken doesn't make any sense. I will keep looking at that

![](https://github.com/DiracSea3921/Project2-StreamCompaction/blob/master/test2.png)

Here is the comparasion of CUDA scatter and Thrust compation, some data may have some problem and I am wondering why my version is a little faster most of the time. 
I didn't use shared memory yet so I will begine with optimize it with shared memory.
