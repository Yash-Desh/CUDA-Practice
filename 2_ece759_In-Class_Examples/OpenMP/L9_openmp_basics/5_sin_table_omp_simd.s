	.file	"5_sin_table_omp_simd.cpp"
	.text
	.section	.rodata
	.align 8
	.type	_ZL3PIE, @object
	.size	_ZL3PIE, 8
_ZL3PIE:
	.long	1413754136
	.long	1074340347
	.text
	.globl	main
	.type	main, @function
main:
.LFB991:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2080, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$256, -2068(%rbp)
	movl	$0, -2072(%rbp)
	jmp	.L2
.L3:
	pxor	%xmm1, %xmm1
	cvtsi2sdl	-2072(%rbp), %xmm1
	movsd	.LC0(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movl	-2072(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rax, -2064(%rbp,%rdx,8)
	addl	$1, -2072(%rbp)
.L2:
	cmpl	$256, -2072(%rbp)
	jl	.L3
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L5
	call	__stack_chk_fail@PLT
.L5:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE991:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	1413754136
	.long	1075388923
	.align 8
.LC1:
	.long	0
	.long	1081081856
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
