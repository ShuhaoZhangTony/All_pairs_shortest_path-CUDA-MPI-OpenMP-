#!/bin/bash
#
# Put your Job commands here.
#
#------------------------------------------------
for Size in 1200 1800 2400 3000 3600 4200 4800; do
	/opt/openmpi/bin/mpirun -np $NSLOTS -machinefile $TMPDIR/machines -mca btl tcp,self,sm \
	/home/team30/LAB2/APSPtest $Size
done
#------------------------------------------------
