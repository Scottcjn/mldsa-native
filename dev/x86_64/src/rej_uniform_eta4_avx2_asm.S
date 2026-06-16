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
#define mask %ymm2
#define eta %ymm3
#define bound %ymm4
#define g0 %xmm5
#define g1 %xmm6

        .text

/*
 * unsigned mld_rej_uniform_eta4_avx2_asm(int32_t *r, const uint8_t *buf,
 *                                        const uint8_t *table)
 *
 * Rejection sampling for ETA=4 polynomial coefficients.
 * Extracts 4-bit nibbles from a byte buffer and accepts those < 9.
 * Output: coefficient = 4 - nibble, producing values in [-4, 4].
 *
 * Arguments:   out (rdi): pointer to output coefficient array r (256 x int32_t)
 *              in  (rsi): pointer to input byte buffer buf (272 bytes)
 *              tab (rdx): pointer to rejection sampling lookup table (256x8 bytes)
 *
 * Returns:     ctr (eax): number of valid coefficients written to r
 */
        .balign 4
        .global MLD_ASM_NAMESPACE(rej_uniform_eta4_avx2_asm)
MLD_ASM_FN_SYMBOL(rej_uniform_eta4_avx2_asm)

// Construct broadcast constants
        movl    $0x0F0F0F0F, good
        vmovd   good, %xmm2
        vpbroadcastd %xmm2, mask              // mask: extract low 4 bits from each byte

        movl    $0x04040404, good
        vmovd   good, %xmm3
        vpbroadcastd %xmm3, eta               // eta: broadcast ETA=4

        movl    $0x09090909, good
        vmovd   good, %xmm4
        vpbroadcastd %xmm4, bound             // bound: rejection threshold (9)

// Initialize counters
        xorl    ctr, ctr                       // ctr = 0
        xorl    pos, pos                       // pos = 0

/*
 * Main SIMD loop: process 16 input bytes into 32 nibbles, producing up to
 * 32 coefficients per iteration (4 sub-iterations of 8 nibbles each).
 *
 * Loop-head guards: ctr <= MLDSA_N - 8 = 248 and pos <= BUFLEN - 16 = 256.
 *
 * Mid-iter early exits at ctr > 248 prevent buffer overshoot: each sub-iter
 * stores 8 ints starting at r[ctr], and ctr advances by popcount(<= 8). With
 * ctr <= 248 entering a sub-iter, the store touches r[248..256] — exactly
 * fitting the 256-int output buffer.
 */
rej_uniform_eta4_avx2_asm_loop:
        cmpl    $248, ctr                      // MLDSA_N - 8
        ja      rej_uniform_eta4_avx2_asm_scalar
        cmpl    $256, pos                      // MLD_AVX2_REJ_UNIFORM_ETA4_BUFLEN - 16
        ja      rej_uniform_eta4_avx2_asm_scalar

        // Load 16 bytes and extract 32 nibbles into f0
        vpmovzxbw (in, %rcx), f0               // load 16 bytes, zero-extend to 16 words
        vpsllw  $4, f0, f1                     // shift left by 4 to align high nibbles
        vpor    f1, f0, f0                     // OR: each word now has nibble duplicated
        vpand   mask, f0, f0                   // mask to get 4 bits per byte

        // Check which nibbles are < 9 and compute eta - nibble
        vpsubb  bound, f0, f1                  // f1 = nibble - 9 (negative if valid)
        vpsubb  f0, eta, f0                    // f0 = 4 - nibble
        vpmovmskb f1, good                     // extract sign bits (valid = 1)

        // Sub-iter 1: extract low 128 bits of f0; process bits 0..7 of mask
        vextracti128 $0, f0, g0
        movzbl  %r8b, %r10d                    // tmp = good & 0xFF
        vmovq   (tab, tmp, 8), g1              // g1 = table[good & 0xFF] (8 byte indices)
        vpshufb g1, g0, g1                     // compact accepted nibbles to front
        vpmovsxbd g1, f1                       // sign-extend 8 bytes -> 8 int32
        vmovdqu f1, (out, %rax, 4)             // store 8 ints at r[ctr]
        popcntl %r10d, cnt                     // cnt = popcount(good & 0xFF)
        addl    cnt, ctr                       // ctr += cnt
        shrl    $8, good                       // shift good for next sub-iter
        addl    $4, pos                        // 4 input bytes consumed

        cmpl    $248, ctr                      // mid-iter exit if ctr > 248
        ja      rej_uniform_eta4_avx2_asm_scalar

        // Sub-iter 2: shift xmm5 down by 8 bytes; process next 8 bits of mask
        vpsrldq $8, g0, g0
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        shrl    $8, good
        addl    $4, pos

        cmpl    $248, ctr
        ja      rej_uniform_eta4_avx2_asm_scalar

        // Sub-iter 3: extract high 128 bits of f0
        vextracti128 $1, f0, g0
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        shrl    $8, good
        addl    $4, pos

        cmpl    $248, ctr
        ja      rej_uniform_eta4_avx2_asm_scalar

        // Sub-iter 4: shift xmm5 down by 8 bytes; process final 8 bits of mask
        vpsrldq $8, g0, g0
        movzbl  %r8b, %r10d
        vmovq   (tab, tmp, 8), g1
        vpshufb g1, g0, g1
        vpmovsxbd g1, f1
        vmovdqu f1, (out, %rax, 4)
        popcntl %r10d, cnt
        addl    cnt, ctr
        addl    $4, pos

        jmp     rej_uniform_eta4_avx2_asm_loop

/*
 * Scalar fallback loop: process remaining bytes one nibble at a time.
 * Each byte contains two 4-bit nibbles (low and high), each accepted iff < 9.
 */
rej_uniform_eta4_avx2_asm_scalar:
        cmpl    $256, ctr                      // MLDSA_N
        jae     rej_uniform_eta4_avx2_asm_done
        cmpl    $272, pos                      // MLD_AVX2_REJ_UNIFORM_ETA4_BUFLEN
        jae     rej_uniform_eta4_avx2_asm_done

        movzbl  (in, %rcx), val                // load 1 byte
        incl    pos

        // Process low nibble
        movl    val, %r10d
        andl    $0x0F, %r10d                   // extract low 4 bits
        cmpl    $9, %r10d
        jae     rej_uniform_eta4_avx2_asm_high_nibble

        movl    $4, cnt
        subl    %r10d, cnt                     // 4 - nibble
        movl    cnt, (out, %rax, 4)
        incl    ctr

        cmpl    $256, ctr
        jae     rej_uniform_eta4_avx2_asm_done

rej_uniform_eta4_avx2_asm_high_nibble:
        // Process high nibble
        shrl    $4, val
        andl    $0x0F, val
        cmpl    $9, val
        jae     rej_uniform_eta4_avx2_asm_scalar

        movl    $4, %r10d
        subl    val, %r10d                     // 4 - nibble
        movl    %r10d, (out, %rax, 4)
        incl    ctr

        jmp     rej_uniform_eta4_avx2_asm_scalar

rej_uniform_eta4_avx2_asm_done:
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
#undef mask
#undef eta
#undef bound
#undef g0
#undef g1

/* simpasm: footer-start */
#endif /* MLD_ARITH_BACKEND_X86_64_DEFAULT && !MLD_CONFIG_MULTILEVEL_NO_SHARED \
        */
