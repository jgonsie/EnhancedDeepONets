Info<< "Reading field T\n" << endl;

volScalarField T
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Initializing velocity field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh//,
    //dimensionedVector("zero", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector::zero) // Initial value
);

Info<< "Creating heterogeneous diffusivity field DT\n" << endl;
volScalarField DT
(
    IOobject
    (
        "DT",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero",  dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.0)  // Initial value
);

// Create IOobject for reading
IOdictionary regionDTDict
(
    IOobject
    (
        "regionDTDict",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

// Read a scalar list
List<scalar> regionDT(regionDTDict.lookup("regionDT"));
// Read a scalar
List<scalar> Uparam(regionDTDict.lookup("Uparam"));

#include "createFvOptions.H"

