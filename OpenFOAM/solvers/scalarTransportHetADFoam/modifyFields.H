//Modify values of DT with a for loop
for (label celli = 0; celli < DT.internalField().size(); ++celli)
{
    if (mesh.C()[celli].component(0) < 0.5)
    {
        DT[celli] = regionDT[0];
    }
    else 
    {
        DT[celli] = regionDT[1];
    }
}

//Modify values of U with a for loop
/* for (label celli = 0; celli < nCells; ++celli)
{
    U[celli][0] = Uparam * mesh.C()[celli].component(0) * (2 * mesh.C()[celli].component(1) - 1) * (mesh.C()[celli].component(0) - 1);
    U[celli][1] = Uparam * mesh.C()[celli].component(1) * (2 * mesh.C()[celli].component(0) - 1) * (mesh.C()[celli].component(1) - 1);
} */
