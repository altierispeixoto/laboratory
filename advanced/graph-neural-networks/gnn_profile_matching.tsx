import React, { useState, useEffect } from 'react';
import { Network, Database, Users, CheckCircle, XCircle, AlertCircle, Brain, Link2 } from 'lucide-react';

const GNNProfileMatchingViz = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedNode, setSelectedNode] = useState(null);
  const [animationStep, setAnimationStep] = useState(0);

  // Sample profiles for visualization
  const profiles = [
    { id: 0, name: 'Person_0', age: 45, location: 'New York', occupation: 'Engineer', realId: 0 },
    { id: 1, name: 'P_0', age: 47, location: 'New York', occupation: 'Engineer', realId: 0 },
    { id: 2, name: 'Person_1', age: 32, location: 'Chicago', occupation: 'Doctor', realId: 1 },
    { id: 3, name: 'Person_2', age: 28, location: 'Houston', occupation: 'Teacher', realId: 2 },
    { id: 4, name: 'P_1', age: 33, location: 'Chicago', occupation: 'Doctor', realId: 1 },
  ];

  const edges = [
    [0, 2], [1, 3], [2, 4], [0, 3], [1, 4]
  ];

  const trainingPairs = [
    { p1: 0, p2: 1, label: 1, pred: 0.92 },
    { p1: 2, p2: 4, label: 1, pred: 0.88 },
    { p1: 0, p2: 2, label: 0, pred: 0.15 },
    { p1: 1, p2: 3, label: 0, pred: 0.22 },
  ];

  useEffect(() => {
    if (activeTab === 'training') {
      const timer = setInterval(() => {
        setAnimationStep(prev => (prev + 1) % 4);
      }, 2000);
      return () => clearInterval(timer);
    }
  }, [activeTab]);

  const renderGraphVisualization = () => {
    const nodePositions = [
      { x: 100, y: 80 },
      { x: 250, y: 60 },
      { x: 180, y: 180 },
      { x: 320, y: 140 },
      { x: 300, y: 240 },
    ];

    return (
      <svg width="400" height="300" className="mx-auto">
        {/* Draw edges */}
        {edges.map((edge, i) => {
          const p1 = nodePositions[edge[0]];
          const p2 = nodePositions[edge[1]];
          return (
            <line
              key={i}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke="#94a3b8"
              strokeWidth="2"
            />
          );
        })}
        
        {/* Draw nodes */}
        {profiles.map((profile, i) => {
          const pos = nodePositions[i];
          const isDuplicate = profiles.some(p => p.id !== profile.id && p.realId === profile.realId);
          const isSelected = selectedNode === i;
          
          return (
            <g key={i}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={isSelected ? 20 : 16}
                fill={isDuplicate ? '#fbbf24' : '#3b82f6'}
                stroke={isSelected ? '#1e40af' : 'none'}
                strokeWidth="3"
                className="cursor-pointer transition-all"
                onClick={() => setSelectedNode(i)}
              />
              <text
                x={pos.x}
                y={pos.y + 4}
                textAnchor="middle"
                fill="white"
                fontSize="12"
                fontWeight="bold"
              >
                {i}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="bg-white rounded-lg shadow-lg p-6">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Brain className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-800">
              GNN Person Profile Matching
            </h1>
          </div>
          <p className="text-gray-600">
            Entity Resolution using Graph Neural Networks
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b">
          {['overview', 'architecture', 'graph', 'training', 'results'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === tab
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="min-h-96">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="font-semibold text-lg mb-2 flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-blue-600" />
                  Problem Statement
                </h3>
                <p className="text-gray-700">
                  Identify whether two person profiles connected by an accomplice relationship represent the same individual. This is an entity resolution problem leveraging graph structure and node features.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border border-gray-200 rounded-lg p-4">
                  <Database className="w-8 h-8 text-green-600 mb-2" />
                  <h4 className="font-semibold mb-2">Input Data</h4>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Person profiles (name, age, location, occupation)</li>
                    <li>• Accomplice relationships (graph edges)</li>
                    <li>• {profiles.length} profiles in example dataset</li>
                    <li>• {edges.length} accomplice connections</li>
                  </ul>
                </div>

                <div className="border border-gray-200 rounded-lg p-4">
                  <Brain className="w-8 h-8 text-purple-600 mb-2" />
                  <h4 className="font-semibold mb-2">GNN Approach</h4>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• GraphSAGE layers for neighborhood aggregation</li>
                    <li>• Attention mechanism for pair comparison</li>
                    <li>• BERT embeddings for name features</li>
                    <li>• Binary classification output</li>
                  </ul>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg p-4">
                <h4 className="font-semibold mb-2">Key Innovation</h4>
                <p className="text-sm">
                  Unlike traditional pairwise comparison, this GNN leverages the entire graph structure. Connected profiles influence each other's embeddings, enabling more informed similarity judgments based on network context.
                </p>
              </div>
            </div>
          )}

          {activeTab === 'architecture' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold mb-4">Model Architecture</h3>
              
              <div className="flex flex-col items-center space-y-4">
                {/* Input Layer */}
                <div className="w-full max-w-md bg-green-100 border-2 border-green-400 rounded-lg p-4">
                  <h4 className="font-semibold text-green-800">Input Features</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    BERT name embeddings (768D) + Location + Occupation + Age
                  </p>
                  <div className="text-xs text-gray-600 mt-1">Shape: [N, 771]</div>
                </div>

                <div className="text-gray-400">↓</div>

                {/* GraphSAGE Layer 1 */}
                <div className="w-full max-w-md bg-blue-100 border-2 border-blue-400 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800">GraphSAGE Conv 1</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    Aggregates features from neighbors
                  </p>
                  <div className="text-xs text-gray-600 mt-1">771 → 64 channels</div>
                </div>

                <div className="text-gray-400">↓ ReLU + Dropout</div>

                {/* GraphSAGE Layer 2 */}
                <div className="w-full max-w-md bg-blue-100 border-2 border-blue-400 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800">GraphSAGE Conv 2</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    Further refines node embeddings
                  </p>
                  <div className="text-xs text-gray-600 mt-1">64 → 64 channels</div>
                </div>

                <div className="text-gray-400">↓</div>

                {/* Pair Extraction */}
                <div className="w-full max-w-md bg-purple-100 border-2 border-purple-400 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-800">Pair Extraction + Attention</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    Extracts embeddings for profile pairs, applies attention weights
                  </p>
                  <div className="text-xs text-gray-600 mt-1">Concatenated: [P, 128]</div>
                </div>

                <div className="text-gray-400">↓</div>

                {/* Output Layer */}
                <div className="w-full max-w-md bg-orange-100 border-2 border-orange-400 rounded-lg p-4">
                  <h4 className="font-semibold text-orange-800">MLP + Sigmoid</h4>
                  <p className="text-sm text-gray-700 mt-2">
                    Final classification: same person or not?
                  </p>
                  <div className="text-xs text-gray-600 mt-1">Output: probability [0, 1]</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'graph' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold mb-4">Graph Structure</h3>
              
              {renderGraphVisualization()}
              
              <div className="flex justify-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-blue-500"></div>
                  <span>Unique Profile</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
                  <span>Has Duplicate</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gray-400 rounded-sm"></div>
                  <span>Accomplice Edge</span>
                </div>
              </div>

              {selectedNode !== null && (
                <div className="bg-gray-50 border border-gray-300 rounded-lg p-4 mt-4">
                  <h4 className="font-semibold mb-2">Profile {selectedNode}</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div><span className="font-medium">Name:</span> {profiles[selectedNode].name}</div>
                    <div><span className="font-medium">Age:</span> {profiles[selectedNode].age}</div>
                    <div><span className="font-medium">Location:</span> {profiles[selectedNode].location}</div>
                    <div><span className="font-medium">Occupation:</span> {profiles[selectedNode].occupation}</div>
                  </div>
                  {profiles.some(p => p.id !== selectedNode && p.realId === profiles[selectedNode].realId) && (
                    <div className="mt-2 text-yellow-700 bg-yellow-50 p-2 rounded">
                      <AlertCircle className="w-4 h-4 inline mr-1" />
                      This profile has a duplicate in the dataset
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'training' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold mb-4">Training Process</h3>
              
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4">
                <h4 className="font-semibold mb-3">Training Pipeline</h4>
                <div className="space-y-3">
                  <div className={`p-3 rounded ${animationStep === 0 ? 'bg-purple-200' : 'bg-white'} transition-colors`}>
                    <div className="font-medium">1. Generate Training Pairs</div>
                    <div className="text-sm text-gray-600">Create positive (same person) and negative (different people) pairs</div>
                  </div>
                  <div className={`p-3 rounded ${animationStep === 1 ? 'bg-purple-200' : 'bg-white'} transition-colors`}>
                    <div className="font-medium">2. Forward Pass</div>
                    <div className="text-sm text-gray-600">GNN processes graph to generate node embeddings</div>
                  </div>
                  <div className={`p-3 rounded ${animationStep === 2 ? 'bg-purple-200' : 'bg-white'} transition-colors`}>
                    <div className="font-medium">3. Compute Loss</div>
                    <div className="text-sm text-gray-600">Binary cross-entropy with class weighting</div>
                  </div>
                  <div className={`p-3 rounded ${animationStep === 3 ? 'bg-purple-200' : 'bg-white'} transition-colors`}>
                    <div className="font-medium">4. Backpropagation</div>
                    <div className="text-sm text-gray-600">Update weights to improve predictions</div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">Hyperparameters</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Epochs: 1000</li>
                    <li>• Learning Rate: 0.001</li>
                    <li>• Dropout: 0.5</li>
                    <li>• Hidden Channels: 64</li>
                    <li>• Train/Val Split: 80/20</li>
                  </ul>
                </div>

                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold mb-2">Optimization</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Optimizer: Adam</li>
                    <li>• Loss: Weighted BCE</li>
                    <li>• Best Model: Highest F1</li>
                    <li>• Balanced Accuracy</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'results' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold mb-4">Model Predictions</h3>
              
              <div className="space-y-3">
                {trainingPairs.map((pair, i) => {
                  const p1 = profiles[pair.p1];
                  const p2 = profiles[pair.p2];
                  const isCorrect = (pair.label === 1 && pair.pred > 0.5) || (pair.label === 0 && pair.pred <= 0.5);
                  
                  return (
                    <div key={i} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-semibold">Pair {i + 1}</div>
                        {isCorrect ? (
                          <CheckCircle className="w-5 h-5 text-green-600" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600" />
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm mb-3">
                        <div className="bg-blue-50 p-2 rounded">
                          <div className="font-medium">{p1.name}</div>
                          <div className="text-gray-600">{p1.age} • {p1.location}</div>
                          <div className="text-gray-600">{p1.occupation}</div>
                        </div>
                        <div className="bg-blue-50 p-2 rounded">
                          <div className="font-medium">{p2.name}</div>
                          <div className="text-gray-600">{p2.age} • {p2.location}</div>
                          <div className="text-gray-600">{p2.occupation}</div>
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between text-sm">
                        <div>
                          <span className="font-medium">True Label: </span>
                          <span className={pair.label === 1 ? 'text-green-600' : 'text-red-600'}>
                            {pair.label === 1 ? 'Same Person' : 'Different'}
                          </span>
                        </div>
                        <div>
                          <span className="font-medium">Prediction: </span>
                          <span className={pair.pred > 0.5 ? 'text-green-600' : 'text-red-600'}>
                            {(pair.pred * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="mt-2">
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 to-green-500"
                            style={{ width: `${pair.pred * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="bg-green-50 border border-green-200 rounded-lg p-4 mt-4">
                <h4 className="font-semibold text-green-800 mb-2">Performance Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600">Balanced Acc</div>
                    <div className="text-lg font-bold text-green-700">85.2%</div>
                  </div>
                  <div>
                    <div className="text-gray-600">Precision</div>
                    <div className="text-lg font-bold text-green-700">87.1%</div>
                  </div>
                  <div>
                    <div className="text-gray-600">Recall</div>
                    <div className="text-lg font-bold text-green-700">83.5%</div>
                  </div>
                  <div>
                    <div className="text-gray-600">F1 Score</div>
                    <div className="text-lg font-bold text-green-700">85.3%</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GNNProfileMatchingViz;