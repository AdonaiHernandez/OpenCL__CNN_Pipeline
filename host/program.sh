service lightdm stop
rmmod -f altvipfb
rmmod cfbfillrect
rmmod cfbimgblt
rmmod cfbcopyarea
aocl program /dev/acl0 pipeconv.aocx
modprobe altvipfb
service lightdm start

