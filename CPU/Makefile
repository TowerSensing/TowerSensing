CXXFLAGS := -O3 -std=c++17 -shared -fPIC $(USER_DEFINES)
BUILD_DIR := ./lib

ifeq ($(USER_DEFINES), -DSIMD_COMPRESS)
    CXXFLAGS += -rdynamic -march=native
endif


all: \
	$(BUILD_DIR)/CMSketch.so \
	$(BUILD_DIR)/CountSketch.so \
	$(BUILD_DIR)/CUSketch.so \
	$(BUILD_DIR)/CMMSketch.so \
	$(BUILD_DIR)/CMLSketch.so \
	$(BUILD_DIR)/CSMSketch.so \
	$(BUILD_DIR)/ClusterReduceWrapper.so \

# The Count sketch uses int32_t counters
$(BUILD_DIR)/CountSketch.so: api.cpp
	@mkdir -p $(BUILD_DIR)
	g++ $(CXXFLAGS) $< -DSKETCH=CountSketch -DCOUNTER_TYPE=int32_t -o $@

# Other sketches use uint32_t counters
$(BUILD_DIR)/%.so: api.cpp
	@mkdir -p $(BUILD_DIR)
	g++ $(CXXFLAGS) $< -DSKETCH=$* -o $@

$(BUILD_DIR)/ClusterReduceWrapper.so: ClusterReduceWrapper.cpp
	@mkdir -p $(BUILD_DIR)
	g++ $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
