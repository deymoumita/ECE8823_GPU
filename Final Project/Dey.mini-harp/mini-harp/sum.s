/*
This program computes the sum of four arrays in parallel uisng four threads in a warp.
*/

.def ARRAY_SIZE 0x8

.perm x
.entry
.global
entry:	ldi %r0, Array1;
        ldi %r1, #1;   
        clone %r1;			/* copy lane 0 registers into lane 1 */
	ldi %r0, Array2;
      	ldi %r1, #2;
        clone %r1;			/* copy lane 0 registers into lane 2 */
	ldi %r0, Array3;
      	ldi %r1, #3;
        clone %r1;			/* copy lane 0 registers into lane 3 */
      	ldi %r0, Array0;
      	ldi %r1, #0;
        ldi %r2, #4;
      	jalis %r7, %r2, sum;		/* parallel call 'sum' on 4 lanes */
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
        st %r3, %r0, #0 ;		/* store result */
        jmprt %r7;               	/* exit parallel call */

.perm r
Array0:  .word -1 -2 -3 -4 -5 -6 -7 -8
Array1:  .word  1  2  3  4  5  6  7  8
Array2:  .word  3  4  5  6  7  8  9  1
Array3:  .word  2  4  5  6  7  9  1  2

.perm rw
Output:  .word 0 0 0 0
