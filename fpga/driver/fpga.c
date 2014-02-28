/* File:   fpga.c       */
/* Author: Matthew Stehr  */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <asm/uaccess.h>   /* copy_*_user */
#include <linux/pci.h>
#include <linux/interrupt.h>
#include <linux/workqueue.h>
#include <linux/delay.h>

MODULE_AUTHOR("Matthew Stehr");
MODULE_LICENSE("GPL");

#define VENDOR_ID (0x1172)
#define DEVICE_ID (0x0004)
#define NUM_DEV   (1)

#define SDRAM   ( 0x00000000 )
#define CRA     ( 0x08000000 )
#define DMA     ( 0x08006000 )
#define SW_IN   ( 0x08006020 )
#define LED_OUT ( 0x08006030 )

#define SDRAM_SIZE ( 128 * 1024 )

/* Function prototypes */
static ssize_t fpga_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t fpga_write(struct file *, const char __user *, size_t, loff_t *);

static int  fpga_probe(struct pci_dev *, const struct pci_device_id *);
static void fpga_remove(struct pci_dev *);

static void fpga_work_handler(struct work_struct *);

DECLARE_WORK( fpga_work, fpga_work_handler );

static struct fpga_dev {
    dev_t dev;
    struct cdev cdev;
    struct class *pclass;
    void __iomem *hw_addr;
} fpga_dev;

static struct file_operations fpga_fops = {
    .owner =    THIS_MODULE,
    .read =     fpga_read,
    .write =    fpga_write,
};

static struct pci_device_id fpga_pci_tbl[] = {
    { PCI_DEVICE(VENDOR_ID, DEVICE_ID) },
    { 0, },
};

MODULE_DEVICE_TABLE(pci, fpga_pci_tbl);

static struct pci_driver fpga_pci_driver = {
    .name = "fpga_pci_driver",
    .id_table = fpga_pci_tbl,
    .probe = fpga_probe,
    .remove = fpga_remove,
};

static irqreturn_t fpga_irq_handler(int irq, void *data)
{
    /* Clear edge detect bits on SW PIO */
    iowrite32( 0xFFFFFFFF, fpga_dev.hw_addr + SW_IN + 0x0C );
    schedule_work( &fpga_work );
    return IRQ_HANDLED;
}

/* tasklets quicker, but must be atomic, wereas workqueues can sleep */
static void fpga_work_handler(struct work_struct *work)
{
    u32 sw;
    sw = ioread32( fpga_dev.hw_addr + SW_IN );
    iowrite32( sw, fpga_dev.hw_addr + LED_OUT );
    mdelay( 500 );
    iowrite32( 0, fpga_dev.hw_addr + LED_OUT );
}

static ssize_t fpga_read(struct file *filp, char __user *buf, size_t count, loff_t *ppos)
{
    u8 byte;
    u32 i;

    if( !fpga_dev.hw_addr ) {
        printk(KERN_INFO "fpga: NULL HW address during Read\n" );
        return 0;
    }

    if( count >= SDRAM_SIZE ) return -EFAULT;

    for( i = 0; i < count; ++i ) {
        byte = ioread8( fpga_dev.hw_addr + SDRAM + i );
        if( put_user( byte, buf + i ) ) return i;
    }
    return count;
}

static ssize_t fpga_write(struct file *filp, const char __user *buf, size_t count, loff_t *ppos)
{
    u8 byte;
    u32 i;

    if( !fpga_dev.hw_addr ) {
        printk(KERN_INFO "fpga: NULL HW address during Write\n" );
        return 0;
    }

    if( count >= SDRAM_SIZE ) return -EFAULT;

    for( i = 0; i < count; ++i ) {
        if( get_user( byte, buf + i ) ) return i;
        iowrite8( byte, fpga_dev.hw_addr + SDRAM + i );
    }
    return count;
}

static int fpga_probe(struct pci_dev *pdev, const struct pci_device_id *pdev_id)
{
    int err;
    printk(KERN_DEBUG "fpga: PCI probe...\n");

    err = pci_enable_device( pdev );
    if( err ) {
        printk(KERN_INFO "fpga: Failed to enable PCI device memory\n");
        return err;
    }

    /* Request BARs be reserved */
    err = pci_request_regions( pdev, fpga_pci_driver.name );
    if( err ) {
        printk(KERN_INFO "fpga: BAR request failed\n");
        pci_disable_device( pdev );
        return err;
    }
    printk(KERN_DEBUG "fpga: BAR request successful\n");

    /* Remap BAR 0 address to virtual kernel address space */
    fpga_dev.hw_addr = ioremap_nocache( pci_resource_start( pdev, 0), pci_resource_len( pdev, 0 ) );
 //   fpga_dev.hw_addr = pci_ioremap_bar( pdev, 0 );
    if( !fpga_dev.hw_addr ) {
        printk(KERN_INFO "fpga: Failed to remap BAR address to kernel space\n");
        pci_release_regions( pdev );
        pci_disable_device( pdev );
        return err;
    }
    printk(KERN_DEBUG "fpga: BAR remap successful\n");

    /* Be greedy and just hold irq here instead of on open() call */
    err = request_irq( pdev->irq, fpga_irq_handler, IRQF_SHARED, "FPGA_int", &fpga_dev );
    if( err ) {
        printk(KERN_DEBUG "fpga: Failed to acquire IRQ...\n");
        iounmap( fpga_dev.hw_addr );
        pci_release_regions( pdev );
        pci_disable_device( pdev );
        return -1;
    }

    /* Be greedy and register DMA channel here too */
    // TODO: Register DMA channel

    /* Enable interrupts on PIO - SW0-SW4 only */
    iowrite32( 0xF, fpga_dev.hw_addr + SW_IN + 0x08 );

    printk(KERN_DEBUG "fpga: PCI probe finished\n");
    return 0;  /* Success */
}

static void fpga_remove(struct pci_dev *pdev)
{
    /* Reverse the setup process... */
    printk(KERN_DEBUG "fpga: Removing PCI device...\n");

    cancel_work_sync( &fpga_work );
    free_irq( pdev->irq, &fpga_dev );
    iounmap( fpga_dev.hw_addr );
    pci_release_regions( pdev );
    pci_disable_device( pdev );
    printk(KERN_DEBUG "fpga: PCI device removed\n");
}

static int __init fpga_init( void )
{
    int err;
    printk(KERN_DEBUG "fpga: Initializing...\n");

    /* Request a major number and NUM_DEV minor numbers starting at 0 */
    err = alloc_chrdev_region( &fpga_dev.dev, 0, NUM_DEV, "fpga" );
    if (err < 0) {
        printk(KERN_WARNING "fpga: Can't allocate a major number\n");
        return err;
    }
    printk(KERN_DEBUG "fpga: Allocated major %d\n", MAJOR( fpga_dev.dev ));

    /* Setup cdev struct */
    cdev_init( &fpga_dev.cdev, &fpga_fops );
    fpga_dev.cdev.owner = THIS_MODULE;

    /* Tell the kernel about the cdev */
    err = cdev_add( &fpga_dev.cdev, fpga_dev.dev, 1 );
    if (err < 0) {  // Failed to add cdev, so unregister (exit wont be called)
        printk(KERN_NOTICE "fpga: Error %d adding fpga\n", err);
        unregister_chrdev_region( fpga_dev.dev, NUM_DEV );
        return err;
    }
    printk(KERN_DEBUG "fpga: Character device added\n");

    /* Register the PCI device */
    err = pci_register_driver( &fpga_pci_driver );
    if (err < 0) {  /* Failed to reg. pci device, so cleanup (exit wont be called) */
        printk(KERN_NOTICE "fpga: Error %d registering pci device\n", err);
        cdev_del( &fpga_dev.cdev );
        unregister_chrdev_region( fpga_dev.dev, NUM_DEV );
        return err;
    }
    printk(KERN_DEBUG "fpga: PCI driver registered\n");

    /* Create class needed to update sysfs */
    fpga_dev.pclass = class_create( THIS_MODULE, "chardrv" );
    if( !fpga_dev.pclass ) {
        printk(KERN_NOTICE "fpga: Error creating device class\n");
        pci_unregister_driver( &fpga_pci_driver );
        cdev_del( &fpga_dev.cdev );
        unregister_chrdev_region( fpga_dev.dev, NUM_DEV );
        return -1;
    }
    printk(KERN_DEBUG "fpga: Device class created\n");

    /* Create the device node */
    if( !device_create( fpga_dev.pclass, NULL, fpga_dev.dev, NULL, "FPGA" ) ) {
        printk(KERN_NOTICE "fpga: Error creating device node\n");
        class_destroy( fpga_dev.pclass );
        pci_unregister_driver( &fpga_pci_driver );
        cdev_del( &fpga_dev.cdev );
        unregister_chrdev_region( fpga_dev.dev, NUM_DEV );
        return -1;
    }
    printk(KERN_DEBUG "fpga: Device node FPGA created\n");

    printk(KERN_DEBUG "fpga: Module initialized\n");
    return 0;  /* Success */
}

static void fpga_exit( void )
{
    printk(KERN_DEBUG "fpga: Exiting\n");

    /* Reverse the setup process... */
    device_destroy( fpga_dev.pclass, fpga_dev.dev );
    class_destroy( fpga_dev.pclass );
    pci_unregister_driver( &fpga_pci_driver );
    cdev_del( &fpga_dev.cdev );
    unregister_chrdev_region( fpga_dev.dev, NUM_DEV );

    printk(KERN_DEBUG "fpga: Succesful cleanup\n");
}

module_init( fpga_init );
module_exit( fpga_exit );
