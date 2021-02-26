/*
Test program to exercise barriers
*/

.def TOKEN      0x80

.perm x
.entry
.global
entry:
        ldi %r2, kernel_w;
        ldi %r0, #0;                    /* set warp id #0 */
        ldi %r1, #1;                    /* set warp id #1 */
        wspawn %r0, %r2, %r0;           /* spawn a new warp #0 */
        wspawn %r0, %r2, %r1;           /* spawn a new warp #1 */

        ldi %r0, TOKEN;
        ldi %r1, Input;
        st %r0, %r1, #0;                /* store input */

        ldi %r0, #2;
        ldi %r1, #3;
        bar %r0, %r1;                   /* barrier #2 - all warps ready to read input */

        ldi %r0, #0;
        ldi %r1, #2;
        bar %r0, %r1;                   /* barrier #0 - warp0 has written output */

        ldi %r1, Output;
        ld %r0, %r1, (__WORD*0);
        jali %r7, printhex;		/* print output 0 */
        jali %r7, printnl;

        ldi %r1, Output;
        ld %r0, %r1, (__WORD*1);
        jali %r7, printhex;		/* print output 1 */
        jali %r7, printnl;

        ldi %r0, #1;
        ldi %r1, #2;
        bar %r0, %r1;                   /* barrier #1 - warp1 has written output */

        ldi %r1, Output;
        ld %r0, %r1, (__WORD*2);
        jali %r7, printhex;		/* print output 2 */
        jali %r7, printnl;

        ldi %r1, Output;
        ld %r0, %r1, (__WORD*3);
        jali %r7, printhex;		/* print output 3 */
        jali %r7, printnl;

        halt;				/* end execution */

.global
kernel_w:
        ldi %r1, #1;                    /* set lane1 index */
        clone %r1;			/* copy lane 0 registers into lane 1 */
        ldi %r1, #0;                    /* set lane0 index */
        ldi %r2, #2;
        jalis %r7, %r2, kernel_p;	/* parallel call 'sum' on 2 lanes */
        halt;				/* end execution */

kernel_p:
        ori %r5, %r0, #0;               /* backup warp id */
        ori %r6, %r1, #0;               /* backup lane id */

        ldi %r0, #2;
        ldi %r1, #3;
        bar %r0, %r1;                   /* 1st barrier */

        ldi %r0, Input;
        ld %r1, %r0, #0;                /* load input */

        muli %r2, %r5, #2;              /* warp offset */
        add %r2, %r2, %r6;              /* add lane offset */
        add %r1, %r1, %r2;              /* result computation */

        ldi %r0, Output;		/* output address */
        muli %r2, %r2, __WORD;          /* word scaling */
        add  %r0, %r0, %r2;             /* add offset */
        st %r1, %r0, #0 		/* store result */

        ldi %r0, #2;
        bar %r5, %r0;                   /* 2nd barrier */

        jmprt %r7;               	/* exit parallel call */

.perm rw
Input:   .word 0

.perm rw
Output:  .word 0 0 0 0
