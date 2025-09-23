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

/*//M odify internal values of U with a for loop over cells
for (label celli = 0; celli < nCells; ++celli)
{
    U[celli][0] = Uparam[0] * mesh.C()[celli].component(0) * (2 * mesh.C()[celli].component(1) - 1) * (mesh.C()[celli].component(0) - 1);
    U[celli][1] = - Uparam[0] * mesh.C()[celli].component(1) * (2 * mesh.C()[celli].component(0) - 1) * (mesh.C()[celli].component(1) - 1);
}

//Modify boundary values of U with a for loop over patches
forAll(U.boundaryField(), patchI)
{
    const fvPatch& patch = mesh.boundary()[patchI];              // fvPatch (field data)
    const polyPatch& geoPatch = mesh.boundaryMesh()[patchI];     // polyPatch (geometry)
    fvPatchVectorField& bf = U.boundaryFieldRef()[patchI];

    label nFaces = patch.size();
    Field<vector> faceValues(nFaces);

    forAll(patch, faceI)
    {
        const point& faceCenter = geoPatch.faceCentres()[faceI];

        scalar x = faceCenter.x();
        scalar y = faceCenter.y();

        vector value(
            Uparam[0] * x * (2 * y - 1) * (x - 1),
            - Uparam[0] * y * (2 * x - 1) * (y - 1),
            0
        );

        faceValues[faceI] = value;
    }

    bf = fixedValueFvPatchVectorField(patch, faceValues);
} */

//Modify internal values of U with a for loop over cells
forAll(U, cellI)
{
    const point& cellCenter = mesh.C()[cellI];

    scalar x = cellCenter.x();
    scalar y = cellCenter.y();

    vector& u = U[cellI];

    // Modify components directly
    //u.x() = Uparam[0] * x * (2 * y - 1) * (x - 1);
    //u.y() = - Uparam[0] * y * (2 * x - 1) * (y - 1);
    u.x() = Uparam[0] * (1 - 2*y);
    u.y() = -Uparam[0] * (1 - 2*x);
    u.z() = 0.0;
}

forAll(U.boundaryField(), patchI)
{
    fvPatchVectorField& bf = U.boundaryFieldRef()[patchI];

    if (bf.type() == "fixedValue")
    {
        const fvPatch& patch = bf.patch();
        const polyPatch& geoPatch = mesh.boundaryMesh()[patchI];

        forAll(patch, faceI)
        {
            const point& faceCenter = geoPatch.faceCentres()[faceI];

            scalar x = faceCenter.x();
            scalar y = faceCenter.y();

            // Access and modify the patch value directly
            vector& u = bf[faceI];

            //u.x() = Uparam[0] * x * (2 * y - 1) * (x - 1);
            //u.y() = - Uparam[0] * y * (2 * x - 1) * (y - 1);
            u.x() = Uparam[0] * (1 - 2*y);
            u.y() = -Uparam[0] * (1 - 2*x);
            u.z() = 0.0;
        }
    }
}
