/*
 * Copyright (c) The mldsa-native project authors
 * SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT
 */

/* References
 * ==========
 *
 * - [REF_AVX2]
 *   CRYSTALS-Dilithium optimized AVX2 implementation
 *   Bai, Ducas, Kiltz, Lepoint, Lyubashevsky, Schwabe, Seiler, Stehlé
 *   https://github.com/pq-crystals/dilithium/tree/master/avx2
 */

/*
 * This file is derived from the public domain
 * AVX2 Dilithium implementation @[REF_AVX2].
 */

#include "../../../common.h"
#if defined(MLD_ARITH_BACKEND_X86_64_DEFAULT) && !defined(MLD_CONFIG_MULTILEVEL_NO_SHARED)
/* simpasm: header-end */

#define out %rdi
#define in %rsi
#define tab %rdx

#define ctr %eax
#define pos %ecx

#define good %r8d
#define cnt %r9d
#define tmp %r10
#define val %r11d

#define f0 %ymm0
#define f1 %ymm1
#define f2 %ymm2
#define mask %ymm3
#define eta %ymm4
#define bound %ymm5
#define v_const %ymm6
#define p_const %ymm7
#define g0 %xmm8
#define g1 %xmm9

        .text

/*
 * unsigned mld_rej_uniform_eta2_avx2_asm(int32_t *r, const uint8_t *buf,
 *                                        const uint8_t *table)
 *
 * Rejection sampling for ETA=2 polynomial coefficients.
 * Extracts 4-bit nibbles from a byte buffer and accepts those < 15.
 * Applies modulo-5 reduction: t = t - (205 * t >> 10) * 5
 * Output: coefficient = 2 - t, producing values in [-2, 2].
 *
 * Arguments:   out (rdi): pointer to output coefficient array r (256 x int32_t)
 *              in  (rsi): pointer to input byte buffer buf
 *              tab (rdx): pointer to rejection sampling lookup table (256x8 bytes)
 *
 * Returns:     ctr (eax): number of valid coefficients written to r
 */
        .balign 4
        .global MLD_ASM_NAMESPACE(rej_uniform_eta2_avx2_asm)
MLD_ASM_FN_SYMBOL(rej_uniform_eta2_avx2_asm)

// Construct broadcast constants
        movl    $0x0F0F0F0F, good
        vmovd   good, %xmm3
        vpbroadcastd %xmm3, mask              // mask: extract low 4 bits from each byte

        movl    $0x02020202, good
        vmovd   good, %xmm4
        vpbroadcastd %xmm4, eta               // eta: broadcast ETA=2

        movl    $0x0F0F0F0F, good
        vmovd   good, %xmm5
        vpbroadcastd %xmm5, bound             // bound: rejection threshold (15)

        // Modulo-5 magic constants
        // v = -6560 == 32*round(-2**10 / 5) for multiply-high-round-scale
        movl    $-6560, good
        vpinsrw $0, good, %xmm6, %xmm6
        vpbroadcastw %xmm6, v_const           // v_const: -6560 for mulhrs

        movl    $5, good
        vpinsrw $0, good, %xmm7, %xmm7
        vpbroadcastw %xmm7, p_const           // p_const: 5 for mullo

// Initialize counters
        xorl    ctr, ctr                       // ctr = 0
        xorl    pos, pos                       // pos = 0

/*
 * Main SIMD loop: process 16 input bytes into 32 nibbles, producing up to 8
 * coefficients per iteration (processing 4 groups of 8 nibbles each).
 * Loops while ctr <= MLDSA_N - 8 and pos <= BUFLEN - 16.
 */
rej_uniform_eta2_avx2_asm_loop:
        cmpl    $248, ctr                      // MLDSA_N - 8
        ja      rej_uniform_eta2_avx2_asm_scalar
        cmpl    $120, pos                      // MLD_AVX2_REJ_UNIFORM_ETA2_BUFLEN - 16
        ja      rej_uniform_eta2_avx2_asm_scalar

        // Load 16 bytes and extract 32 nibbles into f0
        vpmovzxbw (in, %rcx), f0               // load 16 bytes, zero-extend to 16 words
        vpsllw  $4, f0, f1                     // shift left by 4 to align high nibbles
        vpor    f1, f0, f0                     // OR: each word now has nibble duplicated
        vpand   mask, f0, f0                   // mask to get 4 bits per byte

        // Check which nibbles are < 15 and compute eta - nibble
        vpsubb  bound, f0, f1                  // f1 = nibble - 15 (negative if valid)
        vpsubb  f0, eta, f0                    // f0 = 2 - nibble
        vpmovmskb f1, good                     // extract sign bits (valid = 1)

        // For valid nibbles, reduce (2 - nibble) mod 5 (centered, range [-2,2]):
        //   t = (2 - nibble) - (205 * (2 - nibble) >> 10) * 5
        // matching r = 2 - (nibble mod 5) from the reference.

        // Process first group of 8 nibbles (low 128 bits, low 64 bits of that)
        vextracti128 $0, f0, g0                // extract low 128 bits
        movzbl  %r8b, %r10d                      // get low 8 bits of mask
        vmovq   (tab, tmp, 8), g1              // load shuffle indices from table
        vpshufb g1, g0, g1                         // compact valid nibbles
        vpmovsxbd g1, f1                       // sign-extend bytes to dwords

        // Apply modulo-5 reduction
        vpmulhrsw v_const, f1, f2              // f2 = mulhrs(f1, -6560)
        vpmullw p_const, f2, f2                // f2 = f2 * 5
        vpaddd  f2, f1, f1                     // f1 = (2 - nibble) reduced mod 5

        vmovdqu f1, (out, %rax, 4)             // store 8 dwords
        popcntl %r10d, cnt                     // popcount of low 8 bits (in r10d)
        addl    cnt, ctr                       // ctr += popcount(low 8 bits)
        shrl    $8, good
        addl    $4, pos                        // consumed 4 input bytes

        cmpl    $248, ctr
        ja      rej_uniform_eta2_avx2_asm_scalar

        // Process second group of 8 nibbles (low 128 bits, high 64 bits)
        vpsrldq $8, g0, g0                         // shift right to get next 8 nibbles
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vpmulhrsw v_const, f1, f2
        vpmullw p_const, f2, f2
        vpaddd  f2, f1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        shrl    $8, good
        addl    $4, pos

        cmpl    $248, ctr
        ja      rej_uniform_eta2_avx2_asm_scalar

        // Process third group of 8 nibbles (high 128 bits, low 64 bits)
        vextracti128 $1, f0, g0                // extract high 128 bits
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vpmulhrsw v_const, f1, f2
        vpmullw p_const, f2, f2
        vpaddd  f2, f1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        shrl    $8, good
        addl    $4, pos

        cmpl    $248, ctr
        ja      rej_uniform_eta2_avx2_asm_scalar

        // Process fourth group of 8 nibbles (high 128 bits, high 64 bits)
        vpsrldq $8, g0, g0
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vpmulhrsw v_const, f1, f2
        vpmullw p_const, f2, f2
        vpaddd  f2, f1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        addl    $4, pos

        jmp     rej_uniform_eta2_avx2_asm_loop

/*
 * Scalar fallback loop: process remaining bytes one nibble at a time.
 * Each byte contains two 4-bit nibbles (low and high).
 * Apply modulo-5 reduction: t = t - (205 * t >> 10) * 5
 */
rej_uniform_eta2_avx2_asm_scalar:
        cmpl    $256, ctr                      // MLDSA_N
        jae     rej_uniform_eta2_avx2_asm_done
        cmpl    $136, pos                      // MLD_AVX2_REJ_UNIFORM_ETA2_BUFLEN
        jae     rej_uniform_eta2_avx2_asm_done

        movzbl  (in, %rcx), val                // load 1 byte
        incl    pos

        // Process low nibble
        movl    val, %r10d
        andl    $0x0F, %r10d                   // extract low 4 bits
        cmpl    $15, %r10d
        jae     rej_uniform_eta2_avx2_asm_high_nibble

        // Apply modulo-5 reduction: t = t - (205 * t >> 10) * 5
        movl    %r10d, val
        imull   $205, val
        shrl    $10, val
        imull   $5, val
        subl    val, %r10d                     // tmp = tmp - (205*tmp>>10)*5

        movl    $2, %r11d
        subl    %r10d, %r11d                   // 2 - tmp
        movl    %r11d, (out, %rax, 4)
        incl    ctr

        cmpl    $256, ctr
        jae     rej_uniform_eta2_avx2_asm_done

rej_uniform_eta2_avx2_asm_high_nibble:
        // Reload original byte for high nibble
        movzbl  -1(in, %rcx), val              // reload byte
        shrl    $4, val
        andl    $0x0F, val
        cmpl    $15, val
        jae     rej_uniform_eta2_avx2_asm_scalar

        // Apply modulo-5 reduction
        movl    val, %r10d
        imull   $205, %r10d
        shrl    $10, %r10d
        imull   $5, %r10d
        subl    %r10d, val                     // val = val - (205*val>>10)*5

        movl    $2, %r10d
        subl    val, %r10d                     // 2 - val
        movl    %r10d, (out, %rax, 4)
        incl    ctr

        jmp     rej_uniform_eta2_avx2_asm_scalar

rej_uniform_eta2_avx2_asm_done:
        ret

/* To facilitate single-compilation-unit (SCU) builds, undefine all macros.
 * Don't modify by hand -- this is auto-generated by scripts/autogen. */
#undef out
#undef in
#undef tab
#undef ctr
#undef pos
#undef good
#undef cnt
#undef tmp
#undef val
#undef f0
#undef f1
#undef f2
#undef mask
#undef eta
#undef bound
#undef v_const
#undef p_const
#undef g0
#undef g1

/* simpasm: footer-start */
#endif /* MLD_ARITH_BACKEND_X86_64_DEFAULT && !MLD_CONFIG_MULTILEVEL_NO_SHARED \
        */
