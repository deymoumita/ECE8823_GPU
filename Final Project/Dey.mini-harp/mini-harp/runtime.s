/* runtime library */

.align 4096
.def CONSOLE_ADDR 0x1ffffc

.perm x

.global
printhex:    ldi %r1, CONSOLE_ADDR;	/* set console address */
	     ldi %r2, (__WORD * 8);	/* word size in bits */
printhex_l1: subi %r2, %r2, #4;
             shr %r3, %r0, %r2;
             andi %r3, %r3, #15;
             subi %r4, %r3, #10;
             isneg @p0, %r4;             
       @p0 ? addi %r3, %r3, #0x30	/* digits */
	     notp @p0, @p0;
       @p0 ? addi %r3, %r4, #0x61	/* letters */
             st %r3, %r1, #0;		/* write character */
             rtop @p0, %r2;
       @p0 ? jmpi printhex_l1;
             jmpr %r7;

.global
prints:      ldi %r1, CONSOLE_ADDR;	/* set console address */
prints_l:    ld   %r2, %r0, #0;		/* load a word from source address */
             andi %r2, %r2, #0xff;	/* mask out byte character */
             iszero @p0, %r2;
       @p0 ? jmpi prints_end;		/* exit if NULL character */
             st %r2, %r1, #0;		/* write character */
             addi %r0, %r0, #1;		/* advance source address to next byte */
             jmpi prints_l;
prints_end:  jmpr %r7;

.global
printnl:   ldi %r1, CONSOLE_ADDR;	/* set console address */
           ldi %r2, #0x0a;
           st %r2, %r1, #0; 		/* write '\n' */
           jmpr %r7;
