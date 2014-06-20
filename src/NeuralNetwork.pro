TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += /opt/local/include/
LIBS += -L/opt/local/lib
LIBS += -lboost_system-mt -lboost_filesystem-mt

SOURCES += main.cpp \
    neuralnetwork.cpp

HEADERS += \
    neuralnetwork.h

