#include <pybind11/pybind11.h>
#include "./SubArray.h"
#include "./AdderTree.h"
#include "./Bus.h"
#include "./Buffer.h"
#include "./HTree.h"
#include "./BitShifter.h"
#include "./MaxPooling.h"
//#include "../neurosim_src/typedef.h"
#include "./DRAM.h"
//#include "./WeightGradientUnit.h"
#include "./Multiplier.h"
#include "./FunctionUnit.h"
#include <pybind11/stl.h>


namespace py = pybind11;

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

PYBIND11_MODULE(NeuroSIM, m) {

    py::enum_<DeviceRoadmap>(m, "DeviceRoadmap")
            .value("HP", DeviceRoadmap::HP )
            .value("LSTP",DeviceRoadmap::LSTP)
            .export_values();
    /*
    py::enum_<TransistorType>(m, "TransistorType")
            .value("conventional", TransistorType::conventional )
            .value("FET_2D",TransistorType::FET_2D)
            .value("TFET",TransistorType::TFET)
            .export_values();

    py::enum_<BusMode>(m, "BusMode")
            .value("HORIZONTAL", BusMode::HORIZONTAL)
            .value("VERTICAL", BusMode::VERTICAL);

    py::enum_<SpikingMode>(m, "SpikingMode")
            .value("NONSPIKING", SpikingMode::NONSPIKING)
            .value("SPIKING", SpikingMode::SPIKING);

    py::enum_<CellAccessType>(m, "CellAccessType")
            .value("CMOS_access", CellAccessType::CMOS_access)
            .value("BJT_access", CellAccessType::BJT_access)
            .value("diode_access", CellAccessType::diode_access)
            .value("none_access", CellAccessType::none_access);

    py::enum_<AreaModify>(m, "AreaModify")
            .value("NONE", AreaModify::NONE)
            .value("MAGIC",AreaModify::MAGIC)
            .value("OVERRIDE",AreaModify::OVERRIDE)
            .export_values();

    py::class_<InputParameter>(m, "InputParameter")
            .def(py::init<>())
            .def_readwrite("temperature", &InputParameter::temperature)
            .def_readwrite("transistorType", &InputParameter::transistorType)
            .def_readwrite("deviceRoadmap", &InputParameter::deviceRoadmap)
            .def_readwrite("processNode", &InputParameter::processNode);

    py::class_<Technology>(m, "Technology")
            .def(py::init<int &, DeviceRoadmap &, TransistorType &>())
            .def_readwrite("featureSize", &Technology::featureSize);



    py::class_<MemCell>(m, "MemCell")
            .def(py::init<>())
            .def_readwrite("widthInFeatureSize", &MemCell::widthInFeatureSize)
            .def_readwrite("heightInFeatureSize", &MemCell::heightInFeatureSize)
            .def_readwrite("featureSize", &MemCell::featureSize)
            .def_readwrite("resistanceOn", &MemCell::resistanceOn)
            .def_readwrite("resistanceOff", &MemCell::resistanceOff)
            .def_readwrite("resistanceAvg", &MemCell::resistanceAvg)
            .def_readwrite("readVoltage", &MemCell::readVoltage)
            .def_readwrite("readPulseWidth", &MemCell::readPulseWidth)
            .def_readwrite("accessVoltage", &MemCell::accessVoltage)
            .def_readwrite("accessType", &MemCell::accessType)
            .def_readwrite("resistanceAccess", &MemCell::resistanceAccess)
            .def_readwrite("maxNumLevelLTP", &MemCell::maxNumLevelLTP)
            .def_readwrite("maxNumLevelLTD", &MemCell::maxNumLevelLTD)
            .def_readwrite("writeVoltage", &MemCell::writeVoltage)
            .def_readwrite("writePulseWidth", &MemCell::writePulseWidth)
            .def_readwrite("nonlinearIV", &MemCell::nonlinearIV)
            .def_readwrite("nonlinearity", &MemCell::nonlinearity);


    py::class_<SubArray>(m, "SubArray")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("CalculateArea", &SubArray::CalculateArea)
            .def("CalculateLatency", &SubArray::CalculateLatency)
            .def("CalculatePower", &SubArray::CalculatePower)
            .def_readwrite("area", &SubArray::area)
            .def_readwrite("usedArea", &SubArray::usedArea)
            .def_readwrite("width", &SubArray::width)
            .def_readwrite("height", &SubArray::height)
            .def_readwrite("readLatency", &SubArray::readLatency)
            //.def_readwrite("readLatencyAG", &SubArray::readLatencyAG)
            .def_readwrite("writeLatency", &SubArray::writeLatency)
            //.def_readwrite("debugmode", &SubArray::debugmode)
            .def_readwrite("readDynamicEnergy", &SubArray::readDynamicEnergy)
            .def_readwrite("writeDynamicEnergy", &SubArray::writeDynamicEnergy);
            //.def_readwrite("readDynamicEnergyAG", &SubArray::readDynamicEnergyAG);
 /
    py::class_<SubArray>(m, "SubArray")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &SubArray::Initialize,
                py::arg("_numRow"),
                py::arg("_numCol"),
                py::arg("_unitWireRes"),
                py::arg("_numColMuxed"),
                py::arg("_levelOutput"),
                py::arg("_numReadPulse"),
                py::arg("_numRowMuxedBP"),
                py::arg("_levelOutputBP"),
                py::arg("_numReadPulseBP"),
                py::arg("_spikingMode"),
                py::arg("clkFreq"))
            .def("CalculateArea", &SubArray::CalculateArea)
            .def("CalculateLatency", &SubArray::CalculateLatency)
            .def("CalculatePower", &SubArray::CalculatePower)
            .def_readwrite("area", &SubArray::area)
            .def_readwrite("usedArea", &SubArray::usedArea)
            .def_readwrite("width", &SubArray::width)
            .def_readwrite("height", &SubArray::height)
            .def_readwrite("readLatency", &SubArray::readLatency)
            //.def_readwrite("readLatencyAG", &SubArray::readLatencyAG)
            .def_readwrite("writeLatency", &SubArray::writeLatency)
            //.def_readwrite("debugmode", &SubArray::debugmode)
            .def_readwrite("readDynamicEnergy", &SubArray::readDynamicEnergy)
            .def_readwrite("writeDynamicEnergy", &SubArray::writeDynamicEnergy);
            //.def_readwrite("readDynamicEnergyAG", &SubArray::readDynamicEnergyAG);
/

    py::class_<Adder>(m, "Adder")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Adder::Initialize)
            .def("CalculateArea", &Adder::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &Adder::CalculateLatency,
                 py::arg("_rampInput"),
                 py::arg("_capLoad"),
                 py::arg("numRead"))
            .def("CalculatePower", &Adder::CalculatePower,
                 py::arg("numRead"),
                 py::arg("numAdderPerOperation"))
            .def_readwrite("area", &Adder::area)
            .def_readwrite("width", &Adder::width)
            .def_readwrite("height", &Adder::height)
            .def_readwrite("numAdder", &Adder::numAdder)
            .def_readwrite("readLatency", &Adder::readLatency)
            .def_readwrite("readDynamicEnergy", &Adder::readDynamicEnergy);

    py::class_<AdderTree>(m, "AdderTree")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &AdderTree::Initialize)
            .def("CalculateArea", &AdderTree::CalculateArea)
            .def("CalculateLatency", &AdderTree::CalculateLatency)
            .def("CalculatePower", &AdderTree::CalculatePower)
            .def_readwrite("area", &AdderTree::area)
            .def_readwrite("width", &AdderTree::width)
            .def_readwrite("height", &AdderTree::height)
            .def_readwrite("readLatency", &AdderTree::readLatency)
            .def_readwrite("readDynamicEnergy", &AdderTree::readDynamicEnergy);

    py::class_<ShiftAdd>(m, "ShiftAdd")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &ShiftAdd::Initialize)
            .def("CalculateArea", &ShiftAdd::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &ShiftAdd::CalculateLatency)
            .def("CalculatePower", &ShiftAdd::CalculatePower)
            .def_readwrite("area", &ShiftAdd::area)
            .def_readwrite("width", &ShiftAdd::width)
            .def_readwrite("height", &ShiftAdd::height)
            .def_readwrite("readLatency", &ShiftAdd::readLatency)
            .def_readwrite("readDynamicEnergy", &ShiftAdd::readDynamicEnergy);

    py::class_<DFF>(m, "DFF")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &DFF::Initialize)
            .def("CalculateArea", &DFF::CalculateArea,
                 py::arg("_newHeight"),
                 py::arg("_newWidth"),
                 py::arg("_option"))
            .def("CalculateLatency", &DFF::CalculateLatency)
            .def("CalculatePower", &DFF::CalculatePower)
            .def_readwrite("area", &DFF::area)
            .def_readwrite("width", &DFF::width)
            .def_readwrite("height", &DFF::height)
            .def_readwrite("readLatency", &DFF::readLatency)
            .def_readonly("numDff", &DFF::numDff)
            .def_readwrite("readDynamicEnergy", &DFF::readDynamicEnergy);
    py::class_<DRAM>(m, "DRAM")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &DRAM::Initialize)
            .def("CalculateLatency", &DRAM::CalculateLatency)
            .def("CalculatePower", &DRAM::CalculatePower)
            .def_readwrite("area", &DRAM::area)
            .def_readwrite("width", &DRAM::width)
            .def_readwrite("height", &DRAM::height)
            .def_readwrite("readLatency", &DRAM::readLatency)
            .def_readwrite("readDynamicEnergy", &DRAM::readDynamicEnergy);
    py::class_<Buffer>(m, "Buffer")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Buffer::Initialize)
            .def("CalculateArea", &Buffer::CalculateArea)
            .def("CalculateLatency", &Buffer::CalculateLatency)
            .def("CalculatePower", &Buffer::CalculatePower)
            .def_readwrite("area", &Buffer::area)
            .def_readwrite("width", &Buffer::width)
            .def_readwrite("height", &Buffer::height)
            .def_readwrite("readLatency", &Buffer::readLatency)
            .def_readwrite("writeLatency", &Buffer::writeLatency)
            .def_readwrite("readDynamicEnergy", &Buffer::readDynamicEnergy)
            .def_readwrite("writeDynamicEnergy", &Buffer::writeDynamicEnergy)
            .def_readwrite("interface_width", &Buffer::interface_width);

    py::class_<Bus>(m, "Bus")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Bus::Initialize)
            .def("CalculateArea", &Bus::CalculateArea,
                 py::arg("foldedratio"),
                 py::arg("overLap"))
            .def("CalculateLatency", &Bus::CalculateLatency)
            .def("CalculatePower", &Bus::CalculatePower)
            .def_readwrite("area", &Bus::area)
            .def_readwrite("width", &Bus::width)
            .def_readwrite("height", &Bus::height)
            .def_readwrite("numRow", &Bus::numRow)
            .def_readwrite("busWidth", &Bus::busWidth)
            .def_readwrite("readLatency", &Bus::readLatency)
            .def_readwrite("readDynamicEnergy", &Bus::readDynamicEnergy);

    py::class_<HTree>(m, "HTree")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &HTree::Initialize)
            .def("CalculateArea", &HTree::CalculateArea)
            .def("CalculateLatency", &HTree::CalculateLatency)
            .def("CalculatePower", &HTree::CalculatePower)
            .def_readwrite("area", &HTree::area)
            .def_readwrite("width", &HTree::width)
            .def_readwrite("height", &HTree::height)
            .def_readwrite("readLatency", &HTree::readLatency)
            .def_readwrite("readDynamicEnergy", &HTree::readDynamicEnergy)
            .def_readwrite("busWidth", &HTree::busWidth);


    py::class_<BitShifter>(m, "BitShifter")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &BitShifter::Initialize)
            .def("CalculateArea", &BitShifter::CalculateArea)
            .def("CalculateLatency", &BitShifter::CalculateLatency)
            .def("CalculatePower", &BitShifter::CalculatePower)
            .def_readwrite("area", &BitShifter::area)
            .def_readwrite("width", &BitShifter::width)
            .def_readwrite("height", &BitShifter::height)
            .def_readwrite("readLatency", &BitShifter::readLatency)
            .def_readwrite("readDynamicEnergy", &BitShifter::readDynamicEnergy);
    py::class_<Multiplier>(m, "Multiplier")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &Multiplier::Initialize)
            .def("CalculateArea", &Multiplier::CalculateArea)
            .def("CalculateLatency", &Multiplier::CalculateLatency)
            .def("CalculatePower", &Multiplier::CalculatePower)
            .def_readwrite("area", &Multiplier::area)
            .def_readwrite("width", &Multiplier::width)
            .def_readwrite("height", &Multiplier::height)
            .def_readwrite("readLatency", &Multiplier::readLatency)
            .def_readwrite("readDynamicEnergy", &Multiplier::readDynamicEnergy);
    py::class_<MaxPooling>(m, "MaxPooling")
            .def(py::init<InputParameter &,Technology &,MemCell &>())
            .def("Configure", &MaxPooling::Initialize)
            .def("CalculateArea", &MaxPooling::CalculateArea)
            .def("CalculateUnitArea", &MaxPooling::CalculateUnitArea)
            .def("CalculateLatency", &MaxPooling::CalculateLatency)
            .def("CalculatePower", &MaxPooling::CalculatePower)
            .def_readwrite("area", &MaxPooling::area)
            .def_readwrite("width", &MaxPooling::width)
            .def_readwrite("height", &MaxPooling::height)
            .def_readwrite("readLatency", &MaxPooling::readLatency)
            .def_readwrite("readDynamicEnergy", &MaxPooling::readDynamicEnergy);
    */
    //py::class_<WeightGradientUnit>(m, "WeightGradientUnit")
    //        .def(py::init<InputParameter &,Technology &,MemCell &>())
    //        .def("Configure", &WeightGradientUnit::Initialize)
    //        .def("CalculateArea", &WeightGradientUnit::CalculateArea)
    //        .def("CalculateLatency", &WeightGradientUnit::CalculateLatency)
    //        .def("CalculatePower", &WeightGradientUnit::CalculatePower)
    //        .def_readwrite("area", &WeightGradientUnit::area)
    //        .def_readwrite("outPrecision", &WeightGradientUnit::outPrecision)
    //        .def_readwrite("width", &WeightGradientUnit::width)
    //        .def_readwrite("height", &WeightGradientUnit::height)
    //        .def_readwrite("readLatency", &WeightGradientUnit::readLatency)
    //        .def_readwrite("readDynamicEnergy", &WeightGradientUnit::readDynamicEnergy);
    //py::class_<SubArray>(m, "SubArray")
    //.def(py::init<InputParameter &,Technology &,MemCell &>());

}