#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

typedef std::vector<float> Vec;
typedef std::vector<Vec> Matrix;

int ceilDiv(int numerator, int denominator) {
    if (denominator <= 0) {
        throw std::invalid_argument("ceilDiv(): denominator must be positive");
    }
    return (numerator + denominator - 1) / denominator;
}

class systolic;

class bigNode {
   private:
    Vec oVect;
    Vec vVect;
    float m;
    float l;
    int queryIndex;
    int keyIndex;
    bigNode* prevNode;
    bigNode* nextNode;
    systolic* attachedScoreNode;
   public:
    int scoreReadyCycle;
    int acceptCycle;
    int upperReadyCycle;
    int latency;
    int initiationInterval;

    explicit bigNode(std::size_t width = 0)
        : oVect(width, 0.0f),
          vVect(width, 0.0f),
          m(-std::numeric_limits<float>::infinity()),
          l(0.0f),
          queryIndex(-1),
          keyIndex(-1),
          prevNode(NULL),
          nextNode(NULL),
          attachedScoreNode(NULL),
          scoreReadyCycle(0),
          acceptCycle(0),
          upperReadyCycle(0),
          latency(3),
          initiationInterval(1) {}

    static void requireSameSize(const Vec& a, const Vec& b, const char* message) {
        if (a.size() != b.size()) {
            throw std::invalid_argument(message);
        }
    }

    static void requireSameWidth(const Matrix& a, const Matrix& b, const char* message) {
        if (a.size() != b.size()) {
            throw std::invalid_argument(message);
        }
        for (std::size_t row = 0; row < a.size(); ++row) {
            requireSameSize(a[row], b[row], message);
        }
    }

    static Vec scaledAdd(const Vec& lhs, float lhsScale, const Vec& rhs, float rhsScale) {
        requireSameSize(lhs, rhs, "scaledAdd(): vector sizes must match");

        Vec out(lhs.size(), 0.0f);
        for (std::size_t laneIndex = 0; laneIndex < lhs.size(); ++laneIndex) {
            out[laneIndex] = lhs[laneIndex] * lhsScale + rhs[laneIndex] * rhsScale;
        }
        return out;
    }

    void reset(std::size_t width) {
        oVect.assign(width, 0.0f);
        vVect.assign(width, 0.0f);
        m = -std::numeric_limits<float>::infinity();
        l = 0.0f;
        queryIndex = -1;
        keyIndex = -1;
        prevNode = NULL;
        nextNode = NULL;
        attachedScoreNode = NULL;
        scoreReadyCycle = 0;
        acceptCycle = 0;
        upperReadyCycle = 0;
    }

    void setQueryIndex(int value) { queryIndex = value; }
    void setKeyIndex(int value) { keyIndex = value; }
    void setPrevNode(bigNode* value) { prevNode = value; }
    void setNextNode(bigNode* value) { nextNode = value; }
    void attachScoreNode(systolic* value) { attachedScoreNode = value; }

    float getMax() const { return m; }
    float getL() const { return l; }
    int getQueryIndex() const { return queryIndex; }
    int getKeyIndex() const { return keyIndex; }
    const Vec& getOutput() const { return oVect; }
    const Vec& getValue() const { return vVect; }
    bigNode* getPrevNode() const { return prevNode; }
    bigNode* getNextNode() const { return nextNode; }
    systolic* getAttachedScoreNode() const { return attachedScoreNode; }

    void setState(float maxValue, float lValue, const Vec& output) {
        m = maxValue;
        l = lValue;
        oVect = output;
    }

    void applyLocal(float score, const Vec& value, int nextKeyIndex) {
        requireSameSize(oVect, value, "applyLocal(): value size must match accumulator size");

        const bool valid = nextKeyIndex <= queryIndex;
        if (!valid) {
            return;
        }

        keyIndex = nextKeyIndex;

        const float mOut = std::max(m, score);
        const float aIn = std::exp(m - mOut);
        const float aSelf = std::exp(score - mOut);
        const float lOut = l * aIn + aSelf;
        const Vec oOut = scaledAdd(oVect, aIn, value, aSelf);

        setState(mOut, lOut, oOut);
        vVect = value;
    }

    Vec normalizedOutput() const {
        if (l == 0.0f) {
            return oVect;
        }

        Vec out = oVect;
        for (std::size_t laneIndex = 0; laneIndex < out.size(); ++laneIndex) {
            out[laneIndex] /= l;
        }
        return out;
    }

    void printState(const char* label) const {
        const Vec normalized = normalizedOutput();

        std::cout << label << "\n";
        std::cout << "  queryIndex = " << queryIndex << "\n";
        std::cout << "  keyIndex = " << keyIndex << "\n";
        std::cout << "  M = " << m << "\n";
        std::cout << "  L = " << l << "\n";
        std::cout << "  O = [";
        for (std::size_t laneIndex = 0; laneIndex < oVect.size(); ++laneIndex) {
            std::cout << oVect[laneIndex];
            if (laneIndex + 1 != oVect.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "  O_final = [";
        for (std::size_t laneIndex = 0; laneIndex < normalized.size(); ++laneIndex) {
            std::cout << normalized[laneIndex];
            if (laneIndex + 1 != normalized.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
};

struct DotLaneEvent {
    int queryIndex;
    int keyIndex;
    int laneIndex;
    float qLane;
    float kLane;
    float partialScore;
    int cycle;

    DotLaneEvent()
        : queryIndex(-1),
          keyIndex(-1),
          laneIndex(-1),
          qLane(0.0f),
          kLane(0.0f),
          partialScore(0.0f),
          cycle(0) {}

    DotLaneEvent(int queryIndexValue, int keyIndexValue, int laneIndexValue, float qLaneValue, float kLaneValue, float partialScoreValue, int cycleValue)
        : queryIndex(queryIndexValue),
          keyIndex(keyIndexValue),
          laneIndex(laneIndexValue),
          qLane(qLaneValue),
          kLane(kLaneValue),
          partialScore(partialScoreValue),
          cycle(cycleValue) {}
};

struct ScoreEvent {
    int queryIndex;
    int keyIndex;
    float score;
    int readyCycle;
    std::vector<DotLaneEvent> laneEvents;

    ScoreEvent()
        : queryIndex(-1),
          keyIndex(-1),
          score(0.0f),
          readyCycle(0),
          laneEvents() {}

    ScoreEvent(int queryIndexValue, int keyIndexValue, float scoreValue, int readyCycleValue, const std::vector<DotLaneEvent>& laneEventsValue)
        : queryIndex(queryIndexValue),
          keyIndex(keyIndexValue),
          score(scoreValue),
          readyCycle(readyCycleValue),
          laneEvents(laneEventsValue) {}
};

class systolic {
   private:
    Vec k;
    int keyIndex;
    systolic* next;

   public:
    float qStream_i;
    float m;
    float l;
    int scoreReadyCycle;
    int acceptCycle;
    int upperReadyCycle;
    int dotLatency;
    int dotII;
    systolic* prev;

    explicit systolic(const Vec& storedK, int keyIndexValue)
        : k(storedK),
          keyIndex(keyIndexValue),
          next(NULL),
          qStream_i(0.0f),
          m(-std::numeric_limits<float>::infinity()),
          l(0.0f),
          scoreReadyCycle(0),
          acceptCycle(0),
          upperReadyCycle(0),
          dotLatency(0),
          dotII(1),
          prev(NULL) {}

    int getKeyIndex() const { return keyIndex; }
    void setPrev(systolic* value) { prev = value; }
    void setNext(systolic* value) { next = value; }
    systolic* getPrev() const { return prev; }
    systolic* getNext() const { return next; }
    const Vec& getK() const { return k; }

    void configureDotTiming(int laneParallelism, int reductionDepth, int dotIIValue) {
        dotLatency = ceilDiv(static_cast<int>(k.size()), laneParallelism) + reductionDepth;
        dotII = dotIIValue;
    }

    float stepLane(float qLane, float kLane, float partialScore) {
        qStream_i = qLane;
        return partialScore + qLane * kLane;
    }

    ScoreEvent runDotStream(
        int queryIndexValue,
        int keyIndexValue,
        const Vec& q,
        int startCycle) {
        bigNode::requireSameSize(q, k, "runDotStream(): size mismatch");

        float partialScore = 0.0f;
        std::vector<DotLaneEvent> laneEvents;
        laneEvents.reserve(q.size());

        for (std::size_t laneIndex = 0; laneIndex < q.size(); ++laneIndex) {
            partialScore = stepLane(q[laneIndex], k[laneIndex], partialScore);
            laneEvents.push_back(DotLaneEvent(
                queryIndexValue,
                keyIndexValue,
                static_cast<int>(laneIndex),
                q[laneIndex],
                k[laneIndex],
                partialScore,
                startCycle + static_cast<int>(laneIndex)));
        }

        return ScoreEvent(
            queryIndexValue,
            keyIndexValue,
            partialScore,
            startCycle + static_cast<int>(q.size()),
            laneEvents);
    }

    void updateTiming(const ScoreEvent& scoreEvent, int prevAcceptCycle, int upperLatency, int upperII) {
        scoreReadyCycle = scoreEvent.readyCycle;
        acceptCycle = std::max(scoreReadyCycle, prevAcceptCycle + upperII);
        upperReadyCycle = acceptCycle + upperLatency;
    }
};

struct TimingStats {
    int activeQKCycles;
    int activeDotIssueCycles;
    int activeUpperCycles;
    int maskedSkippedCycles;
    int idleCycles;
    int finalCycle;
    int totalCycleCount;
    float qkUtilization;
    float upperUtilization;
    bool hasExampleScoreEvent;
    ScoreEvent exampleScoreEvent;
    std::vector<ScoreEvent> exampleScoreSequence;

    TimingStats()
        : activeQKCycles(0),
          activeDotIssueCycles(0),
          activeUpperCycles(0),
          maskedSkippedCycles(0),
          idleCycles(0),
          finalCycle(0),
          totalCycleCount(0),
          qkUtilization(0.0f),
          upperUtilization(0.0f),
          hasExampleScoreEvent(false),
          exampleScoreEvent(),
          exampleScoreSequence() {}
};

struct TimingConfig {
    int laneParallelism;
    int reductionDepth;
    int upperVectorParallelism;
    int upperScalarCycles;
    int latency;
    int initiationInterval;

    TimingConfig()
        : laneParallelism(1),
          reductionDepth(0),
          upperVectorParallelism(1),
          upperScalarCycles(3),
          latency(3),
          initiationInterval(1) {}
};

struct AttentionFabric {
    std::vector<bigNode> bigNodes;
    std::vector<systolic> scoreNodes;

    AttentionFabric(const Matrix& kMatrix, std::size_t outputWidth)
        : bigNodes(),
          scoreNodes() {
        bigNodes.reserve(kMatrix.size());
        scoreNodes.reserve(kMatrix.size());

        for (std::size_t nodeIndex = 0; nodeIndex < kMatrix.size(); ++nodeIndex) {
            bigNodes.push_back(bigNode(outputWidth));
            bigNodes.back().setQueryIndex(static_cast<int>(nodeIndex));
        }

        for (std::size_t nodeIndex = 0; nodeIndex < kMatrix.size(); ++nodeIndex) {
            scoreNodes.push_back(systolic(kMatrix[nodeIndex], static_cast<int>(nodeIndex)));
        }

        for (std::size_t nodeIndex = 0; nodeIndex < bigNodes.size(); ++nodeIndex) {
            if (nodeIndex > 0) {
                bigNodes[nodeIndex].setPrevNode(&bigNodes[nodeIndex - 1]);
            }
            if (nodeIndex + 1 < bigNodes.size()) {
                bigNodes[nodeIndex].setNextNode(&bigNodes[nodeIndex + 1]);
            }
            bigNodes[nodeIndex].attachScoreNode(&scoreNodes[nodeIndex]);
        }

        for (std::size_t nodeIndex = 0; nodeIndex < scoreNodes.size(); ++nodeIndex) {
            if (nodeIndex > 0) {
                scoreNodes[nodeIndex].setPrev(&scoreNodes[nodeIndex - 1]);
            }
            if (nodeIndex + 1 < scoreNodes.size()) {
                scoreNodes[nodeIndex].setNext(&scoreNodes[nodeIndex + 1]);
            }
        }
    }
};

float dotProduct(const Vec& q, const Vec& k) {
    bigNode::requireSameSize(q, k, "dotProduct(): size mismatch");

    float score = 0.0f;
    for (std::size_t laneIndex = 0; laneIndex < q.size(); ++laneIndex) {
        score += q[laneIndex] * k[laneIndex];
    }
    return score;
}

Matrix referenceAttention(const Matrix& qMatrix, const Matrix& kMatrix, const Matrix& vMatrix) {
    bigNode::requireSameWidth(qMatrix, kMatrix, "referenceAttention(): Q and K shapes must match");
    if (kMatrix.size() != vMatrix.size()) {
        throw std::invalid_argument("referenceAttention(): K and V row counts must match");
    }

    AttentionFabric fabric(kMatrix, vMatrix.front().size());
    Matrix outputs;
    outputs.reserve(qMatrix.size());

    for (std::size_t queryIndex = 0; queryIndex < qMatrix.size(); ++queryIndex) {
        bigNode& state = fabric.bigNodes[queryIndex];
        state.reset(vMatrix.front().size());
        state.setQueryIndex(static_cast<int>(queryIndex));

        for (std::size_t keyIndex = 0; keyIndex <= queryIndex && keyIndex < kMatrix.size(); ++keyIndex) {
            const float score = dotProduct(qMatrix[queryIndex], kMatrix[keyIndex]);
            state.applyLocal(score, vMatrix[keyIndex], static_cast<int>(keyIndex));
        }

        outputs.push_back(state.normalizedOutput());
    }

    return outputs;
}

Matrix baselineSoftmaxAttention(const Matrix& qMatrix, const Matrix& kMatrix, const Matrix& vMatrix) {
    bigNode::requireSameWidth(qMatrix, kMatrix, "baselineSoftmaxAttention(): Q and K shapes must match");
    if (kMatrix.size() != vMatrix.size()) {
        throw std::invalid_argument("baselineSoftmaxAttention(): K and V row counts must match");
    }

    Matrix outputs;
    outputs.reserve(qMatrix.size());

    for (std::size_t queryIndex = 0; queryIndex < qMatrix.size(); ++queryIndex) {
        Vec scores;
        scores.reserve(queryIndex + 1);

        float m = -std::numeric_limits<float>::infinity();
        for (std::size_t keyIndex = 0; keyIndex <= queryIndex && keyIndex < kMatrix.size(); ++keyIndex) {
            const float score = dotProduct(qMatrix[queryIndex], kMatrix[keyIndex]);
            scores.push_back(score);
            m = std::max(m, score);
        }

        float denom = 0.0f;
        for (std::size_t scoreIndex = 0; scoreIndex < scores.size(); ++scoreIndex) {
            denom += std::exp(scores[scoreIndex] - m);
        }

        Vec out(vMatrix.front().size(), 0.0f);
        for (std::size_t keyIndex = 0; keyIndex < scores.size(); ++keyIndex) {
            const float weight = std::exp(scores[keyIndex] - m) / denom;
            for (std::size_t laneIndex = 0; laneIndex < out.size(); ++laneIndex) {
                out[laneIndex] += weight * vMatrix[keyIndex][laneIndex];
            }
        }

        outputs.push_back(out);
    }

    return outputs;
}

Matrix traditionalMQAVerifier(const Matrix& qMatrix, const Matrix& kMatrix, const Matrix& vMatrix) {
    bigNode::requireSameWidth(qMatrix, kMatrix, "traditionalMQAVerifier(): Q and K shapes must match");
    if (kMatrix.size() != vMatrix.size()) {
        throw std::invalid_argument("traditionalMQAVerifier(): K and V row counts must match");
    }

    Matrix outputs;
    outputs.reserve(qMatrix.size());

    for (std::size_t queryIndex = 0; queryIndex < qMatrix.size(); ++queryIndex) {
        Vec scores;
        scores.reserve(queryIndex + 1);

        for (std::size_t keyIndex = 0; keyIndex <= queryIndex && keyIndex < kMatrix.size(); ++keyIndex) {
            scores.push_back(dotProduct(qMatrix[queryIndex], kMatrix[keyIndex]));
        }

        float maxScore = -std::numeric_limits<float>::infinity();
        for (std::size_t scoreIndex = 0; scoreIndex < scores.size(); ++scoreIndex) {
            maxScore = std::max(maxScore, scores[scoreIndex]);
        }

        float normalizer = 0.0f;
        for (std::size_t scoreIndex = 0; scoreIndex < scores.size(); ++scoreIndex) {
            normalizer += std::exp(scores[scoreIndex] - maxScore);
        }

        Vec output(vMatrix.front().size(), 0.0f);
        for (std::size_t keyIndex = 0; keyIndex < scores.size(); ++keyIndex) {
            const float weight = std::exp(scores[keyIndex] - maxScore) / normalizer;
            for (std::size_t laneIndex = 0; laneIndex < output.size(); ++laneIndex) {
                output[laneIndex] += weight * vMatrix[keyIndex][laneIndex];
            }
        }

        outputs.push_back(output);
    }

    return outputs;
}

TimingStats timingModel(const Matrix& qMatrix, const Matrix& kMatrix, const Matrix& vMatrix, const TimingConfig& config) {
    bigNode::requireSameWidth(qMatrix, kMatrix, "timingModel(): Q and K shapes must match");
    if (kMatrix.size() != vMatrix.size()) {
        throw std::invalid_argument("timingModel(): K and V row counts must match");
    }

    AttentionFabric fabric(kMatrix, vMatrix.front().size());
    for (std::size_t keyIndex = 0; keyIndex < fabric.scoreNodes.size(); ++keyIndex) {
        fabric.scoreNodes[keyIndex].configureDotTiming(config.laneParallelism, config.reductionDepth, config.initiationInterval);
    }

    TimingStats stats;
    int globalLastUpperAcceptCycle = -1;
    int activeScores = 0;
    int lastDotIssueCycle = -1;
    const int upperVectorCycles = ceilDiv(static_cast<int>(vMatrix.front().size()), config.upperVectorParallelism);
    const int upperWorkCyclesPerScore = config.upperScalarCycles + (3 * upperVectorCycles);

    for (std::size_t queryIndex = 0; queryIndex < qMatrix.size(); ++queryIndex) {
        const int queryStartCycle = static_cast<int>(queryIndex);
        int prevAcceptCycle = globalLastUpperAcceptCycle;

        for (std::size_t keyIndex = 0; keyIndex < kMatrix.size(); ++keyIndex) {
            if (keyIndex > queryIndex) {
                ++stats.maskedSkippedCycles;
                continue;
            }

            const int keyOffset = static_cast<int>(keyIndex);
            const ScoreEvent scoreEvent = fabric.scoreNodes[keyIndex].runDotStream(
                static_cast<int>(queryIndex),
                static_cast<int>(keyIndex),
                qMatrix[queryIndex],
                queryStartCycle + keyOffset);
            if (!stats.hasExampleScoreEvent) {
                stats.hasExampleScoreEvent = true;
                stats.exampleScoreEvent = scoreEvent;
            }
            if (stats.exampleScoreSequence.size() < 6) {
                stats.exampleScoreSequence.push_back(scoreEvent);
            }
            fabric.scoreNodes[keyIndex].updateTiming(
                scoreEvent,
                prevAcceptCycle,
                config.latency,
                config.initiationInterval);

            const int dotIssueCycle = queryStartCycle + keyOffset;
            if (dotIssueCycle > lastDotIssueCycle) {
                ++stats.activeDotIssueCycles;
                lastDotIssueCycle = dotIssueCycle;
            }

            ++activeScores;
            stats.activeQKCycles += static_cast<int>(scoreEvent.laneEvents.size());
            stats.activeUpperCycles += upperWorkCyclesPerScore;
            prevAcceptCycle = fabric.scoreNodes[keyIndex].acceptCycle;
            stats.finalCycle = std::max(stats.finalCycle, fabric.scoreNodes[keyIndex].upperReadyCycle);
        }

        globalLastUpperAcceptCycle = prevAcceptCycle;
    }

    stats.idleCycles = std::max(0, stats.finalCycle - stats.activeDotIssueCycles);
    if (stats.finalCycle > 0) {
        stats.qkUtilization = static_cast<float>(stats.activeDotIssueCycles) / static_cast<float>(stats.finalCycle);
        stats.upperUtilization = static_cast<float>(activeScores * config.initiationInterval) / static_cast<float>(stats.finalCycle);
    }
    stats.totalCycleCount = stats.finalCycle + 1;

    return stats;
}

float maxAbsDiff(const Matrix& lhs, const Matrix& rhs) {
    bigNode::requireSameWidth(lhs, rhs, "maxAbsDiff(): shape mismatch");

    float diff = 0.0f;
    for (std::size_t row = 0; row < lhs.size(); ++row) {
        for (std::size_t laneIndex = 0; laneIndex < lhs[row].size(); ++laneIndex) {
            diff = std::max(diff, std::fabs(lhs[row][laneIndex] - rhs[row][laneIndex]));
        }
    }
    return diff;
}

void printMatrix(const Matrix& matrix, const char* label, std::size_t maxRows) {
    std::cout << label << "\n";
    const std::size_t rowsToPrint = std::min(matrix.size(), maxRows);
    for (std::size_t row = 0; row < rowsToPrint; ++row) {
        std::cout << "  row " << row << " = [";
        for (std::size_t laneIndex = 0; laneIndex < matrix[row].size(); ++laneIndex) {
            std::cout << matrix[row][laneIndex];
            if (laneIndex + 1 != matrix[row].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    if (rowsToPrint < matrix.size()) {
        std::cout << "  ... (" << (matrix.size() - rowsToPrint) << " more rows)\n";
    }
}

void printScoreEventTrace(const ScoreEvent& scoreEvent, const char* label) {
    std::cout << label << "\n";
    std::cout << "  queryIndex = " << scoreEvent.queryIndex << "\n";
    std::cout << "  keyIndex = " << scoreEvent.keyIndex << "\n";
    for (std::size_t eventIndex = 0; eventIndex < scoreEvent.laneEvents.size(); ++eventIndex) {
        const DotLaneEvent& laneEvent = scoreEvent.laneEvents[eventIndex];
        std::cout << "  cycle " << laneEvent.cycle
                  << ": Q(" << laneEvent.queryIndex << "," << laneEvent.laneIndex << ") * "
                  << "K(" << laneEvent.keyIndex << "," << laneEvent.laneIndex << ") = "
                  << laneEvent.qLane << " * " << laneEvent.kLane
                  << ", partialScore = " << laneEvent.partialScore << "\n";
    }
    std::cout << "  emitted score = " << scoreEvent.score << "\n";
    std::cout << "  readyCycle = " << scoreEvent.readyCycle << "\n";
}

void printScoreSequencePreview(const std::vector<ScoreEvent>& scoreEvents, const char* label) {
    std::cout << label << "\n";
    for (std::size_t eventIndex = 0; eventIndex < scoreEvents.size(); ++eventIndex) {
        const ScoreEvent& scoreEvent = scoreEvents[eventIndex];
        std::cout << "  event " << eventIndex
                  << ": Q" << scoreEvent.queryIndex
                  << " dot K" << scoreEvent.keyIndex
                  << " -> score " << scoreEvent.score
                  << ", readyCycle = " << scoreEvent.readyCycle << "\n";
    }
}

void printFabricSummary(const AttentionFabric& fabric, const char* label) {
    std::cout << label << "\n";
    std::cout << "  bigNodes = " << fabric.bigNodes.size() << "\n";
    std::cout << "  scoreNodes = " << fabric.scoreNodes.size() << "\n";
    if (!fabric.bigNodes.empty()) {
        std::cout << "  bigNode[0].prev = " << (fabric.bigNodes[0].getPrevNode() == NULL ? "null" : "set") << "\n";
        std::cout << "  bigNode[0].next = " << (fabric.bigNodes[0].getNextNode() == NULL ? "null" : "set") << "\n";
        std::cout << "  bigNode[last].next = " << (fabric.bigNodes.back().getNextNode() == NULL ? "null" : "set") << "\n";
        std::cout << "  bigNode[0].attachedScoreNode = " << (fabric.bigNodes[0].getAttachedScoreNode() == NULL ? "null" : "set") << "\n";
    }
    if (!fabric.scoreNodes.empty()) {
        std::cout << "  scoreNode[0].prev = " << (fabric.scoreNodes[0].getPrev() == NULL ? "null" : "set") << "\n";
        std::cout << "  scoreNode[0].next = " << (fabric.scoreNodes[0].getNext() == NULL ? "null" : "set") << "\n";
        std::cout << "  scoreNode[last].next = " << (fabric.scoreNodes.back().getNext() == NULL ? "null" : "set") << "\n";
    }
}

Matrix generateMatrix(std::size_t rows, std::size_t cols, float rowScale, float colScale, float bias) {
    Matrix out;
    out.reserve(rows);

    for (std::size_t row = 0; row < rows; ++row) {
        Vec values;
        values.reserve(cols);
        for (std::size_t col = 0; col < cols; ++col) {
            const int centeredRow = static_cast<int>(row % 17) - 8;
            const int centeredCol = static_cast<int>(col % 7) - 3;
            const float rowTerm = rowScale * static_cast<float>(centeredRow);
            const float colTerm = colScale * static_cast<float>(centeredCol);
            const float mixTerm = 0.01f * static_cast<float>((row * (col + 1)) % 13);
            values.push_back(rowTerm + colTerm + mixTerm + bias);
        }
        out.push_back(values);
    }

    return out;
}

int main() {
    const std::size_t numQueries = 1024;
    const std::size_t numKeys = 1024;
    const std::size_t numValues = 1024;
    const std::size_t d_k = 16;
    const std::size_t d_v = 16;

    Matrix qMatrix = generateMatrix(numQueries, d_k, 0.02f, 0.03f, 0.10f);
    Matrix kMatrix = generateMatrix(numKeys, d_k, -0.015f, 0.025f, -0.05f);
    Matrix vMatrix = generateMatrix(numValues, d_v, 0.01f, -0.02f, 0.20f);
    AttentionFabric fabric(kMatrix, d_v);

    TimingConfig timingConfig;
    timingConfig.laneParallelism = 1;
    timingConfig.reductionDepth = 0;
    timingConfig.upperVectorParallelism = 1;
    timingConfig.upperScalarCycles = 3;
    timingConfig.latency = 3;
    timingConfig.initiationInterval = 1;

    const Matrix onlineOutputs = referenceAttention(qMatrix, kMatrix, vMatrix);
    const Matrix traditionalVerifierOutputs = traditionalMQAVerifier(qMatrix, kMatrix, vMatrix);
    const Matrix baselineOutputs = baselineSoftmaxAttention(qMatrix, kMatrix, vMatrix);
    const TimingStats stats = timingModel(qMatrix, kMatrix, vMatrix, timingConfig);
    const float epsilon = 1e-4f;
    const float baselineError = maxAbsDiff(onlineOutputs, baselineOutputs);
    const float traditionalVerifierError = maxAbsDiff(onlineOutputs, traditionalVerifierOutputs);
    const bool traditionalMatchesPipelined = traditionalVerifierError <= epsilon;
    const bool baselineMatchesPipelined = baselineError <= epsilon;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "numQueries = " << numQueries << "\n";
    std::cout << "numKeys = " << numKeys << "\n";
    std::cout << "numValues = " << numValues << "\n";
    std::cout << "d_k = " << d_k << "\n";
    std::cout << "d_v = " << d_v << "\n";
    printFabricSummary(fabric, "fabric()");
    printMatrix(onlineOutputs, "referenceAttention()", 3);
    printMatrix(traditionalVerifierOutputs, "traditionalMQAVerifier()", 3);
    printMatrix(baselineOutputs, "baselineSoftmaxAttention()", 3);
    std::cout << "traditionalMQAOutputVectorsMatchPipelined = "
              << (traditionalMatchesPipelined ? "true" : "false") << "\n";
    std::cout << "maxAbsDiffVsTraditionalMQA = " << traditionalVerifierError << "\n";
    std::cout << "matchesTraditionalMQA = " << (traditionalMatchesPipelined ? "true" : "false") << "\n";
    std::cout << "baselineOutputVectorsMatchPipelined = "
              << (baselineMatchesPipelined ? "true" : "false") << "\n";
    std::cout << "maxAbsDiffVsBaseline = " << baselineError << "\n";
    std::cout << "matchesBaseline = " << (baselineMatchesPipelined ? "true" : "false") << "\n";
    printScoreSequencePreview(stats.exampleScoreSequence, "causalScoreOrderPreview()");
    if (stats.hasExampleScoreEvent) {
        printScoreEventTrace(stats.exampleScoreEvent, "exampleScoreStream()");
    }
    std::cout << "timingModel()\n";
    std::cout << "  activeQKCycles = " << stats.activeQKCycles << "\n";
    std::cout << "  activeDotIssueCycles = " << stats.activeDotIssueCycles << "\n";
    std::cout << "  activeUpperCycles = " << stats.activeUpperCycles << "\n";
    std::cout << "  maskedSkippedCycles = " << stats.maskedSkippedCycles << "\n";
    std::cout << "  idleCycles = " << stats.idleCycles << "\n";
    std::cout << "  finalCycle = " << stats.finalCycle << "\n";
    std::cout << "  totalCycleCountFromQ0Lane0ToLastOutput = " << stats.totalCycleCount << "\n";
    std::cout << "  qkUtilization = " << stats.qkUtilization << "\n";
    std::cout << "  upperUtilization = " << stats.upperUtilization << "\n";

    return 0;
}
