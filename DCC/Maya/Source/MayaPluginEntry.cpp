#include <maya/MFnPlugin.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>

MStatus initializePlugin(MObject plugin_object) {
    const MFnPlugin plugin(plugin_object, "Blackframe", "0.1.0", "Any");
    return MS::kSuccess;
}

MStatus uninitializePlugin(MObject plugin_object) {
    const MFnPlugin plugin(plugin_object);
    return MS::kSuccess;
}
