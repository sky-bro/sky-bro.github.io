
## Get a windows iso {#get-a-windows-iso}

[MS: Download Windows 11](https://www.microsoft.com/en-in/software-download/windows11)


## Create new VM with virt-manager {#create-new-vm-with-virt-manager}

open virt-manager

make sure you've connected to the QEMU/KVM (click the File option, then 'Add Connection', make sure hypervisor is selected to QEMU/KVM, and click connect)

now QEMU/KVM will show up that you can add a vm to:

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-create-a-new-virtual-machine.png" caption="<span class=\"figure-number\">Figure 1: </span>create a new virtual machine with virt-manager" >}}


### walk through basic options {#walk-through-basic-options}


#### select Local install media (ISO image or CDROM) {#select-local-install-media--iso-image-or-cdrom}

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-select-windows-iso.png" caption="<span class=\"figure-number\">Figure 2: </span>select windows 11 iso" >}}


#### Configure Memory and CPU {#configure-memory-and-cpu}

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-memory-and-cpu.png" caption="<span class=\"figure-number\">Figure 3: </span>configure memory and cpu" >}}


#### Create a virtual hard disk {#create-a-virtual-hard-disk}

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-create-virtual-hard-disk.png" caption="<span class=\"figure-number\">Figure 4: </span>create a disk" >}}


#### Set VM name, Network, etc. {#set-vm-name-network-etc-dot}

and make sure you select the option: Customize configuration before install.

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-name-network-customize.png" caption="<span class=\"figure-number\">Figure 5: </span>Name, Network, Customize" >}}


### configure hardware {#configure-hardware}

if you've selected the "Customize configuration before install" option, you'll be lead to this hardware configuration prompt.


#### hard disk bus type {#hard-disk-bus-type}

-   Click on SATA Disk 1.
-   Choose the disk bus as VirtIO

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-set-disk-bus-type-as-virtio.png" caption="<span class=\"figure-number\">Figure 6: </span>set disk bus tpe to VirtIO" >}}


#### network device model {#network-device-model}

also set network device model to virtio

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-network-device-model.png" caption="<span class=\"figure-number\">Figure 7: </span>set network device model to virtio" >}}


#### add virtio driver {#add-virtio-driver}

-   click on Add Hardware
-   select storage, click on manage, and attach the virtio driver you've downloaded
-   choose device type as CDROM

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-attach-virtio-driver-iso.png" caption="<span class=\"figure-number\">Figure 8: </span>add virtio driver" >}}


#### change boot order {#change-boot-order}

make sure CDROM 1 is checked and at top.

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-change-boot-order.png" caption="<span class=\"figure-number\">Figure 9: </span>change boot order" >}}


#### (optional) enable TPM {#optional--enable-tpm}

Click on Add  Hardware, Add the TPM as below.

Model – You will see two models, choose TIS,
Backend – select Backend as Emulated.
Version – 2.0

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-add-TPM.png" caption="<span class=\"figure-number\">Figure 10: </span>add TPM" >}}


#### (optional) enable Secure Boot {#optional--enable-secure-boot}

you need to install ovmf and swtpm

```shell
sudo pacman -Sy swtpm ed2k-ovmf
```

{{< figure src="/images/posts/windows-11-vm-with-qemu-kvm/virt-manager-enable-secure-boot.png" caption="<span class=\"figure-number\">Figure 11: </span>enable secure boot" >}}


## Begin installation {#begin-installation}


### bypass checks {#bypass-checks}

Click Begin Installation button on the top left corner to start the installation process, then install Windows like you would on a normal PC.

If you haven't enabled TPM 2.0 and secure boot, you'll not meet the installation requirements of windows 11. But you can bypass these checks.

open command prompt with `Shift+F10`

```bat
REG ADD HKLM\SYSTEM\Setup\Labconfig /v BypassTPMCheck /t REG_DWORD /d 1
REG ADD HKLM\SYSTEM\Setup\Labconfig /v BypassSecureBootCheck /t REG_DWORD /d 1
```

Similarly, you can disable other checks with: BypassSecureCPUCheck, BypassSecureRAMCheck, BypassSecureStorageCheck


### Load driver {#load-driver}

You won't be able to find the virtio hard disk that you have added, click on Load driver. In the prompt, choose windows 11 driver, and click on Next.


### Skip connecting to network {#skip-connecting-to-network}

we don't have the corresponding virtio driver yet, we'll install it in the next section.


## After installation {#after-installation}


### Install VirtIO Drivers {#install-virtio-drivers}

-   Open the Windows Explorer and navigate to the CD-ROM drive.
-   Simply execute (double-click on) virtio-win-gt-x64
-   (optional) Use the virtio-win-guest-tools wizard to install the QEMU Guest Agent and the SPICE agent for an improved remote-viewer experience.
-   (optional) Reboot VM


### remove CDROMs {#remove-cdroms}

-   Remove the windows installer iso after intallation.
-   Keep the virtio iso.


### file sharing {#file-sharing}

recommend sharing files between host and windows guest with samba.


## resources {#resources}

-   [How to properly configure Virt-Manager (QEMU/KVM) with Windows guest](https://askubuntu.com/questions/1146441/how-to-properly-configure-virt-manager-qemu-kvm-with-windows-guest)
