# Curvature Prediction Using Machine Learning
Mentors: Dr. Rahul Babu Koneru, Dr. Bhargav Sriram Siddani,
Undergraduate Researcher: Saman Marandi

  Aircrafts operating in dusty environments suffer from structural damage to the gas turbine engine (GTE) components due to the ingestion of sand and other particulate matter. This leads to higher operational costs and in some tragic cases, loss of life. Particulate separators at the intake keep out particles larger than 80 µm in diameter from entering the hot-section of the GTE. Smaller particles which pass through, melt in the combustor and cause structural damage to the thermal barrier coatings (TBCs) which offer protection against high thermal loads on the gas turbine blades. The molten sand particles, which are a mixture of Calcium, Magnesium, Aluminum, Silica oxides along with trace amounts of other compounds and are referred to as CMAS. The molten CMAS deposits, chemically reacts and eventually infiltrates the TBC leading to altering the thermal properties of the TBC and thus reducing its life. 
To develop mitigation strategies against CMAS attack, it is important to investigate the dynamic wetting of the molten CMAS on a surface coated with TBC. A surface with higher wettability has a larger surface area of contact with the molten droplet increasing its chance of failure. A volume of fluid (VOF) framework, commonly employed to mimic the flow of immiscible fluids, is a useful numerical tool to study surface wettability. In a VOF method, one way to compute the surface tension at the interface is done by using the continuum surface force (CSF) model. The resulting surface tension force is given as F=σκn where, σ is the surface tension, κ is the interface curvature and n is the interface normal vector. An accurate computation of κ is essential for interface reconstruction. One of the techniques used to compute curvature is the height function (HF) approach which uses the volume fraction from a local stencil. A neural-network based surrogate model is developed to compute the curvature as a function of local volume fraction.
