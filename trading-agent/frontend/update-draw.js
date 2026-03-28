const fs = require('fs');
let code = fs.readFileSync('components/TradingChart.tsx', 'utf8');

code = code.replace(
    /const \[activeTool, setActiveTool\] = useState\("pointer"\);/,
    'const [activeTool, setActiveTool] = useState("pointer");\n    const [drawings, setDrawings] = useState<any[]>([]);\n    const [isDrawing, setIsDrawing] = useState(false);\n    const [currentDrawing, setCurrentDrawing] = useState<any>(null);'
);

const drawingHandlers = \
    const handleDrawingStart = (e: React.MouseEvent) => {
        if (activeTool === 'pointer' || activeTool === 'indicators') return; 
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (activeTool === 'text') {
            const txt = prompt("Enter chart text:");
            if (txt) setDrawings([...drawings, { id: Date.now(), type: 'text', x1: x, y1: y, txt }]);
            setActiveTool('pointer');
            return;
        }

        setIsDrawing(true);
        setCurrentDrawing({ id: Date.now(), type: activeTool, x1: x, y1: y, x2: x, y2: y, path: [{x, y}] });
    };

    const handleDrawingMove = (e: React.MouseEvent) => {
        if (!isDrawing || !currentDrawing) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (currentDrawing.type === 'pencil' || currentDrawing.type === 'patterns') {
            setCurrentDrawing({ ...currentDrawing, path: [...currentDrawing.path, {x, y}] });
        } else {
            setCurrentDrawing({ ...currentDrawing, x2: x, y2: y });
        }
    };

    const handleDrawingEnd = () => {
        if (isDrawing && currentDrawing) {
            setDrawings([...drawings, currentDrawing]);
            setIsDrawing(false);
            setCurrentDrawing(null);
            if (activeTool !== 'pencil' && activeTool !== 'patterns') setActiveTool('pointer');
        }
    };
\;

code = code.replace(
    /const handleSearchDate = \(\) => \{/,
    drawingHandlers + '\n\n    const handleSearchDate = () => {'
);


code = code.replace(
    /if\(tool.id === "patterns" \|\| tool.id === "pencil"\) alert\("Advanced drawing tools require TradingView license."\);/g,
    \// Native custom drawing enabled\
);

const newDOM = \
            <div className="flex flex-1 relative" style={{ minHeight: '450px'}}>
                <div className="flex flex-col items-center py-2 space-y-1.5 w-10 border-r border-[#171717] bg-[#0A0A0A] z-10 shrink-0">
                    {tools.map(tool => (
                        <button
                            key={tool.id}
                            onClick={() => setActiveTool(tool.id)}
                            className={\\\p-2 rounded-md transition-colors \\\\\\}
                        >
                            {tool.icon}
                        </button>
                    ))}
                    <button onClick={() => setDrawings([])} className="p-2 rounded-md text-xs text-rose-500 hover:bg-[#171717] transition-all mt-4 font-bold" title="Clear Drawings">?</button>
                </div>

                <div
                    className="flex-1 relative cursor-crosshair"
                    onMouseDown={activeTool !== 'pointer' ? handleDrawingStart : undefined}
                    onMouseMove={activeTool !== 'pointer' ? handleDrawingMove : undefined}
                    onMouseUp={activeTool !== 'pointer' ? handleDrawingEnd : undefined}
                    onMouseLeave={activeTool !== 'pointer' ? handleDrawingEnd : undefined}
                >
                    <div ref={chartContainerRef} className={\\\bsolute inset-0 \\\\\\} />
                    
                    {/* CUSTOM FREE DRAWING LAYER */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none z-40">
                        {drawings.map(d => {
                            if (d.type === 'line') return <line key={d.id} x1={d.x1} y1={d.y1} x2={d.x2} y2={d.y2} stroke="#38BDF8" strokeWidth="2" />;      
                            if (d.type === 'pencil' || d.type === 'patterns') {
                                const dPath = d.path.map((p, i) => (i === 0 ? \\\M \\\ \\\\\\ : \\\L \\\ \\\\\\)).join(' ');        
                                return <path key={d.id} d={dPath} fill="transparent" stroke={d.type === 'pencil' ? "#FCD34D" : "#34d399"} strokeWidth="2" />;
                            }
                            if (d.type === 'text') return <text key={d.id} x={d.x1} y={d.y1} fill="#ffffff" fontSize="14" fontFamily="monospace">{d.txt}</text>;
                            return null;
                        })}
                        {currentDrawing && currentDrawing.type === 'line' && <line x1={currentDrawing.x1} y1={currentDrawing.y1} x2={currentDrawing.x2} y2={currentDrawing.y2} stroke="#38BDF8" strokeWidth="2" strokeDasharray="4" />}      
                        {currentDrawing && (currentDrawing.type === 'pencil' || currentDrawing.type === 'patterns') && (
                            <path d={currentDrawing.path.map((p, i) => (i === 0 ? \\\M \\\ \\\\\\ : \\\L \\\ \\\\\\)).join(' ')} fill="transparent" stroke={currentDrawing.type === 'pencil' ? "#FCD34D" : "#34d399"} strokeWidth="2" />
                        )}
                    </svg>
                </div>
            </div>
\;

code = code.replace(/<div className="flex flex-1 relative" style=\{\{ minHeight: '450px'\}\}>[\s\S]*?<\/div>\n        <\/div>\n    \);\n}/, newDOM + '\n        </div>\n    );\n}');

fs.writeFileSync('components/TradingChart.tsx', code);
console.log('Update Complete');
