ifeq ($(ARCH_NAME),$(filter $(ARCH_NAME),wormhole wormhole_b0))
ERISC_MAKE = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc
ERISC_MAKE_CLEAN = TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/erisc clean
else
ERISC_MAKE = @echo 'Skipping Erisc build for Grayskull.'
ERISC_MAKE_CLEAN = @echo 'Skipping Erisc clean for Grayskull.'
endif

.PHONY: src/firmware
src/firmware: $(TT_METAL_HOME)/src/ckernels
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/brisc
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/ncrisc
	$(ERISC_MAKE)

src/firmware/clean:
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/brisc clean
	TT_METAL_HOME=$(TT_METAL_HOME) $(MAKE) -C src/firmware/riscv/targets/ncrisc clean
	$(ERISC_MAKE_CLEAN)
