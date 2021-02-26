/*
This program computes the sum of four arrays in parallel using four single threaded warps.
use a barrier to ensure that all warps are done with their computation.
warp3 does the final copy of the results to the console output.
*/

.def ARRAY_SIZE 0x8

.perm x
.entry
.global
entry:	ldi %r0, sum_w;
        ldi %r1, #3;
        wspawn %r1, %r0, %r1;
        ldi %r1, #2;
        wspawn %r1, %r0, %r1;
        ldi %r1, #1;
        wspawn %r1, %r0, %r1;
        ldi %r1, #0;
        jmpi sum_w;

.global
sum_w:	iszero @p0, %r1;
  @p0 ? ldi %r0, Array0;                /* warp0 uses Array0 */
        subi %r2, %r1, #1;
        iszero @p0, %r2;
  @p0 ? ldi %r0, Array1;                /* warp1 uses Array1 */
        subi %r2, %r1, #2;
        iszero @p0, %r2;
  @p0 ? ldi %r0, Array2;                /* warp2 uses Array2 */
        subi %r2, %r1, #3;
        iszero @p0, %r2;
  @p0 ? ldi %r0, Array3;                /* warp3 uses Array3 */
        ori %r6, %r1, #0;               /* backup warpid into r6 */
        jali %r7, sum;		        /* execute 'sum' on single thread */
        ldi %r0, #0;
        ldi %r1, #4;
        bar %r0, %r1;                   /* insert barrier0 for 4 warps */
        subi %r1, %r6, #3;
        rtop @p0, %r1;
  @p0 ? halt;                           /* terminate other warps except warp3 */
        ldi %r1, Output;
        ld %r0, %r1, (__WORD*0);
        jali %r7, printhex;		/* print output 0 */
        jali %r7, printnl;
        ldi %r1, Output;
        ld %r0, %r1, (__WORD*1);
        jali %r7, printhex;		/* print output 1 */
        jali %r7, printnl;
        ldi %r1, Output;
        ld %r0, %r1, (__WORD*2);
        jali %r7, printhex;		/* print output 2 */
        jali %r7, printnl;
        ldi %r1, Output;
        ld %r0, %r1, (__WORD*3);
        jali %r7, printhex;		/* print output 3 */
        jali %r7, printnl;
        halt;				/* end execution */

sum:    ldi %r3, #0;  			/* initialize sum to zero */
        ldi %r4, ARRAY_SIZE; 		/* initialize loop counter */

loop:   ld  %r2, %r0, #0;
        add %r3, %r3, %r2;
        addi %r0, %r0, __WORD;  	/* advance array offset */
        subi %r4, %r4, #1;      	/* update loop counter */
        rtop @p0, %r4;      		/* set predicate register */
  @p0 ? jmpi loop;               	/* conditional branch */
        ldi %r0, Output;		/* get output address */
        shli %r1, %r1, #2;
        add %r0, %r0, %r1;		/* add output offset */
        st %r3, %r0, #0 		/* store result */
        jmprt %r7;               	/* exit parallel call */

.perm r
Array0:  .word -1 -2 -3 -4 -5 -6 -7 -8
Array1:  .word  1  2  3  4  5  6  7  8
Array2:  .word  3  4  5  6  7  8  9  1
Array3:  .word  2  4  5  6  7  9  1  2

.perm rw
Output:  .word 0 0 0 0
