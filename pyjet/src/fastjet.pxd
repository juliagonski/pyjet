from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "fastjet.h":
    cdef void raise_py_error()

cdef extern from "fastjet.h" namespace "fastjet::contrib":
    #This could be not right yet
    cdef cppclass Nsubjettiness:
        Nsubjettiness()
        Nsubjettiness(int,
                 const AxesDefinition&,
                 const MeasureDefinition&)

    cdef cppclass NsubjettinessRatio:
        NsubjettinessRatio()
        NsubjettinessRatio(int,int,
                 const AxesDefinition&,
                 const MeasureDefinition&)
    
    cdef cppclass AxesDefinition:
        AxesDefinition() except +
    cdef cppclass KT_Axes(AxesDefinition):
        KT_Axes() except +

    cdef cppclass MeasureDefinition:
        MeasureDefinition(double, double) except +
    cdef cppclass NormalizedMeasure(MeasureDefinition):
        NormalizedMeasure(double, double) except + 

    cdef cppclass EnergyCorrelatorC2:
        EnergyCorrelatorC2()
        EnergyCorrelatorC2(double, Measure, Strategy)
        double result(PseudoJet&)

    cdef cppclass EnergyCorrelatorD2:
        EnergyCorrelatorD2()
        EnergyCorrelatorD2(double, Measure, Strategy)
        double result(PseudoJet&)

    cdef cppclass EnergyCorrelator:
        EnergyCorrelator() 
        EnergyCorrelator(unsigned int,
                         double,
                         Measure,
                         Strategy) except +raise_py_error
        double result(PseudoJet&)

cdef extern from "fastjet.h" namespace "fastjet::contrib::EnergyCorrelator":
    cdef enum Measure "fastjet::contrib::EnergyCorrelator::Measure":
         pt_R,     #//< use transverse momenta and boost-invariant angles,
            #///< eg \f$\mathrm{ECF}(2,\beta) = \sum_{i<j} p_{ti} p_{tj} \Delta R_{ij}^{\beta} \f$
         E_theta,  #///  use energies and angles,
            #///  eg \f$\mathrm{ECF}(2,\beta) = \sum_{i<j} E_{i} E_{j}   \theta_{ij}^{\beta} \f$
         E_inv #///  use energies and invariant mass,
            #///  eg \f$\mathrm{ECF}(2,\beta) = \sum_{i<j} E_{i} E_{j}   (\frac{2 p_{i} \cdot p_{j}}{E_{i} E_{j}})^{\beta/2} \f$

    cdef enum Strategy "fastjet::contrib::EnergyCorrelator::Strategy":
        slow,         # ///< interparticle angles are not cached.
            #///< For N>=3 this leads to many expensive recomputations,
            #///< but has only O(n) memory usage for n particles
        storage_array  #/// the interparticle angles are cached. This gives a significant speed
            #/// improvement for N>=3, but has a memory requirement of (4n^2) bytes.


cdef extern from "fastjet.h" namespace "fastjet":

    cdef cppclass PseudoJet:
        PseudoJet(const double, const double, const double, const double)
        PseudoJet()
        vector[PseudoJet] constituents()
        double E()
        double Et()
        double e()
        double px()
        double py()
        double pz()
        double phi_std()
        double rapidity()
        double pseudorapidity()
        double perp()
        double m()
        int cluster_hist_index()
        double mperp()
        double kt_distance(PseudoJet&)
        double plain_distance(PseudoJet&)
        double squared_distance(PseudoJet&)
        double delta_R(PseudoJet&)
        double delta_phi_to(PseudoJet&)
        double beam_distance()
        bool contains(PseudoJet&)
        bool has_valid_cluster_sequence()
        ClusterSequence* associated_cluster_sequence()
        bool has_constituents()
        void set_user_info(UserInfoBase*)
        bool has_user_info()
        bool has_child(PseudoJet& child)
        bool has_parents(PseudoJet& parent1, PseudoJet& parent2)
        UserInfoBase* user_info_ptr()
        bool has_area()
        double area()
        double area_error()

    vector[PseudoJet] sorted_by_pt(vector[PseudoJet]& jets)

    cdef cppclass ClusterSequence:
        ClusterSequence(vector[PseudoJet]&, JetDefinition&) except +raise_py_error
        vector[PseudoJet] inclusive_jets(double ptmin)
        vector[PseudoJet] exclusive_jets(int njets)
        vector[PseudoJet] exclusive_jets_dcut(double dcut)
        int n_exclusive_jets(double dcut)
        void delete_self_when_unused()
        vector[PseudoJet] unclustered_particles()
        vector[PseudoJet] childless_pseudojets()

    cdef enum JetAlgorithm "fastjet::JetAlgorithm":
        kt_algorithm,
        cambridge_algorithm,
        antikt_algorithm,
        genkt_algorithm,
        cambridge_for_passive_algorithm,
        genkt_for_passive_algorithm,
        ee_kt_algorithm,
        ee_genkt_algorithm,
        plugin_algorithm,
        undefined_jet_algorithm

    cdef cppclass JetDefinition:
        JetDefinition(JetAlgorithm) except +raise_py_error
        JetDefinition(JetAlgorithm, double R) except +raise_py_error
        JetDefinition(JetAlgorithm, double R, double extra) except +raise_py_error

    cdef enum AreaType "fastjet::AreaType":
        invalid_area,
        active_area,
        active_area_explicit_ghosts,
        one_ghost_passive_area,
        passive_area,
        voronoi_area

    cdef cppclass AreaDefinition:
        AreaDefinition(AreaType)
        AreaDefinition()

    cdef cppclass ClusterSequenceArea(ClusterSequence):
        ClusterSequenceArea(vector[PseudoJet]&, JetDefinition&, AreaDefinition&) except +raise_py_error

    cdef cppclass SharedPtr[T]:
        pass


cdef extern from "fastjet.h" namespace "fastjet::PseudoJet":
    cdef cppclass UserInfoBase:
        pass


cdef extern from "fastjet.h":
    cdef bint _USING_EXTERNAL_FASTJET
    void silence()
    bool jet_has_area(PseudoJet*)
    double jet_area(PseudoJet*)
    double jet_area_error(PseudoJet*)


