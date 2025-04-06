import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HeroSection from './components/HeroSection';
import Login from './components/Login';
import Signup from './components/Signup';
import Main from './components/Main';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<HeroSection />} />
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />
                <Route path="/assessment" element={<Main />} />
            </Routes>
        </Router>
    );
}

export default App;