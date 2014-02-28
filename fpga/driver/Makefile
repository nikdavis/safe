# Written by Matthew Stehr
# 01/06/2014

ifneq ($(KERNELRELEASE),)
	obj-m := fpga.o

else
	KERNELDIR ?= /lib/modules/3.11.0-15-generic/build
	PWD := $(shell pwd)

default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

endif


