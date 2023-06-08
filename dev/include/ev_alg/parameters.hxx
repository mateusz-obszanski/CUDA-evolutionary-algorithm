#pragma once
#include "./ea_utils.hxx"
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>

/// parameters needed during runtime
struct EvAlgParams {
    const CostMx       costMx;
    const unsigned int populationSize;
    const unsigned int nLocations;
    const unsigned int nIslands;
    const unsigned int iterationsPerEpoch;
    const unsigned int nGenes;

    EvAlgParams() = delete;
    [[nodiscard]] EvAlgParams(const CostMx costMx, unsigned int populationSize,
                              unsigned int nLocations, unsigned int nIslands,
                              unsigned int iterationsPerEpoch,
                              unsigned int nGenes) noexcept
    : costMx(costMx),
      populationSize(populationSize),
      nLocations(nLocations),
      nIslands(nIslands),
      iterationsPerEpoch(iterationsPerEpoch),
      nGenes(nGenes) {}
};

struct ParamsParseError : public std::exception {
    const std::size_t lineno;
    const std::string line;
    const std::string msg;

    ParamsParseError() = delete;
    ParamsParseError(std::size_t lineno, std::string line)
    : lineno{lineno},
      line(line),
      msg("ParamParseError (line " + std::to_string(lineno) + "): " + line) {}
    ParamsParseError(ParamsParseError const&) = default;
    ParamsParseError(ParamsParseError&&)      = default;

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

struct KeyOverwriteError : public std::exception {
    const std::string key;
    const std::string msg;

    KeyOverwriteError() = delete;

    KeyOverwriteError(std::string key)
    : key(key), msg("KeyOverwriteError: " + key) {}

    KeyOverwriteError(KeyOverwriteError const&) = default;
    KeyOverwriteError(KeyOverwriteError&&)      = default;

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

struct KeyError : public std::exception {
    const std::string key;
    const std::string msg;

    KeyError() = delete;

    KeyError(std::string key) : key(key), msg("KeyError: " + key) {}

    KeyError(KeyError const&) = default;
    KeyError(KeyError&&)      = default;

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

struct ParameterConversionError : public std::exception {
    const std::string key;
    const std::string value;
    const std::string msg;

    ParameterConversionError() = delete;

    ParameterConversionError(std::string key, std::string value)
    : key(key),
      value(value),
      msg("ParameterConversionError: " + key + " (value: " + value + ")") {}

    ParameterConversionError(ParameterConversionError const&) = default;
    ParameterConversionError(ParameterConversionError&&)      = default;

    const char*
    what() const noexcept override {
        return msg.c_str();
    }
};

inline std::unordered_map<std::string, std::string>
parse_params(std::istream& input) {
    std::unordered_map<std::string, std::string> parameterMap;

    std::string line;

    for (std::size_t lineno{0}; std::getline(input, line); ++lineno) {

        trim(line);

        if (line.empty() or line.starts_with('#'))
            continue;

        std::istringstream lineStream(line);

        std::string paramName, value;

        if (not(lineStream >> paramName >> value))
            throw ParamsParseError(lineno, line);

        const bool alreadyContains =
            parameterMap.find(paramName) != parameterMap.end();

        if (alreadyContains)
            throw KeyOverwriteError(paramName);

        parameterMap[paramName] = value;
    }

    return parameterMap;
}

template <typename T>
inline T
convert_parameter(
    std::string const&                                  name,
    std::unordered_map<std::string, std::string> const& parameterMap,
    std::optional<T> const&                             defaultVal = {}) {

    const auto value_ptr = parameterMap.find(name);

    if (value_ptr == parameterMap.end()) {
        if (defaultVal.has_value())
            return defaultVal.value();

        throw KeyError(name);
    }

    const auto value = value_ptr->second;

    std::istringstream converter(value);
    T                  result;

    if (not(converter >> result))
        throw ParameterConversionError(name, value);

    return result;
}

/// parameters needed during setup
struct AllParams {
    const std::size_t  prng_seed;
    const unsigned int iterationsPerEpoch;
    const unsigned int nEpochs;
    const unsigned int nEpochsWithoutImprovement;
    const unsigned int nLocations;
    const float        minCost;
    const float        maxCost;
    const unsigned int islandPopulation; // must be an even number
    const unsigned int nIslands;
    const unsigned int nMigrants;
    // in normal, same-size coding, the first and last (0) location is implicit.
    // If special solution coder is used, this will be different
    const unsigned int nGenes;
    const float        mutationChance;
    const float        migrationRatio;

    [[nodiscard]] AllParams(
        std::size_t prng_seed, unsigned int iterationsPerEpoch,
        unsigned int nEpochs, unsigned int nEpochsWithoutImprovement,
        unsigned int nLocations, unsigned int islandPopulation,
        unsigned int nIslands, float mutationChance, float migrationRatio,
        float minCost = 1e-2f, float maxCost = 1e1f, unsigned int nGenes = 0)
    : prng_seed(prng_seed),
      iterationsPerEpoch(iterationsPerEpoch),
      nEpochs(nEpochs),
      nEpochsWithoutImprovement(nEpochsWithoutImprovement),
      nLocations(nLocations),
      minCost(minCost),
      maxCost(maxCost),
      islandPopulation(islandPopulation),
      nIslands(nIslands),
      nMigrants(static_cast<unsigned int>(
          std::max(1.0f, migrationRatio * islandPopulation))),
      nGenes(nGenes == 0 ? nLocations - 1 : nGenes),
      mutationChance(mutationChance),
      migrationRatio(migrationRatio) {}

    void
    print() const {
        const auto flags = std::cout.flags();

        // set pretty flags
        std::cout << std::scientific << std::boolalpha;

        std::cout << "prng_seed: " << prng_seed << ",\n"
                  << "iterationsPerEpoch: " << iterationsPerEpoch << ",\n"
                  << "nEpochs: " << nEpochs << ",\n"
                  << "nEpochsWithoutImprovement: " << nEpochsWithoutImprovement
                  << ",\n"
                  << "nLocations: " << nLocations << ",\n"
                  << "minCost: " << minCost << ",\n"
                  << "maxCost: " << maxCost << ",\n"
                  << "islandPopulation: " << islandPopulation << ",\n"
                  << "nIslands: " << nIslands << ",\n"
                  << "nMigrants: " << nMigrants << ",\n"
                  << "nGenes: " << nGenes << ",\n"
                  << "mutationChance: " << mutationChance << ",\n"
                  << "migrationRatio: " << migrationRatio << "\n";

        // restoring initial state
        std::cout.flags(flags);
    }

    [[nodiscard]] inline static AllParams
    from_file(std::filesystem::path path) {
        std::ifstream input(path.string());
        const auto    parameterMap = parse_params(input);

        return AllParams(
            convert_parameter<std::size_t>("prng_seed", parameterMap),
            convert_parameter<unsigned int>("iterationsPerEpoch", parameterMap),
            convert_parameter<unsigned int>("nEpochs", parameterMap),
            convert_parameter<unsigned int>("nEpochsWithoutImprovement",
                                            parameterMap),
            convert_parameter<unsigned int>("nLocations", parameterMap),
            convert_parameter<unsigned int>("islandPopulation", parameterMap),
            convert_parameter<unsigned int>("nIslands", parameterMap),
            convert_parameter<float>("mutationChance", parameterMap),
            convert_parameter<float>("migrationRatio", parameterMap),
            convert_parameter<float>("minCost", parameterMap, 1e-2f),
            convert_parameter<float>("maxCost", parameterMap, 1e1f),
            convert_parameter<unsigned int>("nGenes", parameterMap, 0));
    }
};
